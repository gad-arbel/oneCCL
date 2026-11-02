/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/* proxy.cpp
 *
 * Host-side proxy thread implementation.
 */

#include "common/env/env.hpp"
#include "common/global/global.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_utils.hpp"
#include "proxy/proxy_func.hpp"

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>

#include <cstdlib>
#include <new>

/* ======== Proxy thread implementation ======== */

ccl_proxy_thread::ccl_proxy_thread() = default;
ccl_proxy_thread::~ccl_proxy_thread() {
    // The owner (ccl_proxy_service::stop) stops the thread and needs stop()'s
    // return to decide what may be freed, so here the thread is already joined
    // and stop() is a no-op. Kept because ~std::thread on a joinable thread
    // terminates. A false return means we detached a live thread that will keep
    // reading this freed object — unpreventable here, so just report it.
    if (!stop()) {
        LOG_ERROR("[proxy] destroyed while its thread was still running and the thread "
                  "did not exit in time; it is now detached and will keep reading this "
                  "freed object. Owner must stop() first and leak on a false return.");
    }
}

void ccl_proxy_thread::start(ccl_cmd_ring_producer* producer,
                             ccl_kernel_lifecycle* lifecycle,
                             ccl_proxy_staging* staging,
                             sycl::queue* queue,
                             ccl_request_ring_consumer* request_consumer) {
    if (m_thread.joinable())
        return;

    auto& env = ccl::global_data::env();
    auto& global_data = ccl::global_data::get();

    size_t cpu_affinity = env.worker_affinity[global_data.get_local_proc_idx()];
    size_t mem_affinity = env.worker_mem_affinity[global_data.get_local_proc_idx()];

    m_cpu_affinity = static_cast<int>(cpu_affinity);
    m_mem_affinity = static_cast<int>(mem_affinity);

    m_producer = producer;
    m_lifecycle = lifecycle;
    m_staging = staging;
    m_queue = queue;
    m_request_consumer = request_consumer;
    m_stop_flag = false;
    m_exited = false;

    m_thread = std::thread(&ccl_proxy_thread::work_loop, this);
    set_thread_affinity(m_thread.native_handle(), m_cpu_affinity);
}

bool ccl_proxy_thread::stop() {
    if (!m_thread.joinable())
        return true;

    // Notify the thread to exit its work loop
    m_stop_flag.store(true, std::memory_order_release);

    // How long stop() waits for the proxy thread to leave an in-flight op before
    // giving up and detaching it. Not a tunable: at finalize no legitimate op can
    // still be in flight, so any wait longer than a generous slack means the thread
    // is stuck in one of the deadline-free waits (ring completion, ATL wait) and
    // will never come back.
    constexpr auto proxy_stop_timeout = std::chrono::seconds(10);
    constexpr auto proxy_stop_poll_interval = std::chrono::milliseconds(1);
    // The flag is only tested at the top of work_loop, so a thread inside
    // process_host_initiated_op reaches no check until its op completes — and
    // none of the waits in there take a deadline. A plain join() would turn a
    // stuck proxy (dead peer) into a silent ccl::finalize hang, so bound the wait
    // and report instead.
    const auto deadline = std::chrono::steady_clock::now() + proxy_stop_timeout;
    while (!m_exited.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG_ERROR("[proxy] thread did not exit within ",
                      std::chrono::duration_cast<std::chrono::seconds>(proxy_stop_timeout).count(),
                      " s; it is stuck in an in-flight op (kernel-start handshake, ring "
                      "completion, or ATL wait). Detaching and skipping teardown: leaking on "
                      "shutdown is preferable to freeing memory it is still using.");
            m_thread.detach();
            return false;
        }
        std::this_thread::sleep_for(proxy_stop_poll_interval);
    }

    // m_exited is set as work_loop's last action, so this cannot block.
    m_thread.join();
    return true;
}

void ccl_proxy_thread::submit(const ccl_proxy_op& op) {
    std::lock_guard<std::mutex> lock(m_host_queue_mutex);
    m_host_op_queue.push(op);
}

/* Hands out the oldest host op, but only once its persistent kernel has started; until then the op
 * stays at the front and this reports an empty queue, so the loop goes on to the device arm.
 *
 * Running it earlier would park this thread in the op's wait_kernel_start(), and that can be a
 * deadlock: submit() happens at enqueue time, so on an in-order queue the op is usually here while
 * kernels ahead of its own are still running - a device-initiated one among them, which only
 * poll_device_op() can finish. Once the kernel has started there is no such hazard left to block
 * on: everything ahead of it has completed, its requests included, and nothing behind it starts
 * before the op returns and its lifecycle guard sends the exit command.
 *
 * That argument assumes proxy kernels form one in-order stream, which the lifecycle's single
 * >= handshake and the one command ring already do. Only the front is examined, so ops still run
 * in submit (generation) order.
 *
 * out_pending reports whether any op is queued, started or not, so the caller treats an op waiting
 * for its kernel as pending work and keeps the backoff hot instead of sleeping through the start. */
bool ccl_proxy_thread::poll_host_op(ccl_proxy_op& out_op, bool& out_pending) {
    std::lock_guard<std::mutex> lock(m_host_queue_mutex);
    out_pending = !m_host_op_queue.empty();
    if (!out_pending ||
        !ccl_kernel_lifecycle::kernel_started(m_lifecycle, m_host_op_queue.front().generation))
        return false;

    out_op = std::move(m_host_op_queue.front());
    m_host_op_queue.pop();
    return true;
}

/* The device-initiated arm, counterpart of poll_host_op(): one sweep of the request ring - post
 * what the device has published, probe what is in flight, never block on a transfer. Returns
 * whether there was anything to do, including "nothing new, but transfers are still outstanding",
 * so the caller keeps sweeping rather than backing off (see request_ring_host.hpp).
 *
 * Unlike the host arm, one sweep is already the whole non-blocking step - the ring is the queue -
 * so this arm always comes back round to the host queue and m_stop_flag. It is also what runs while
 * a queued host op waits for its kernel to start (see poll_host_op()).
 *
 * What is *not* bounded is the device side: a kernel spinning on a completion after this thread
 * exits hangs the GPU, so stop() must not be reached with requests outstanding. The caller
 * guarantees that by waiting on the kernel's event before finalize. */
bool ccl_proxy_thread::poll_device_op() {
    return m_request_consumer && m_request_consumer->progress();
}

void ccl_proxy_thread::work_loop() {
    // Report the resolved CPU/NUMA placement
    LOG_DEBUG("[proxy] thread: cpu: ",
              m_cpu_affinity,
              ", numa: ",
              ccl::global_data::get().hwloc_wrapper->get_numa_node(m_mem_affinity).to_string());
    // set mem affinity
    ccl::global_data::get().hwloc_wrapper->membind_thread(m_mem_affinity);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::set_thread_name("ccl_proxy_thread");
#endif // CCL_ENABLE_ITT

    ccl_proxy_backoff backoff;
    while (!m_stop_flag.load(std::memory_order_relaxed)) {
        // poll host-initiated operations whose kernel has started
        ccl_proxy_op host_op;
        bool host_op_pending = false;
        if (poll_host_op(host_op, host_op_pending)) {
            // The handler is what makes the stack unwind at all: without one,
            // terminate runs with the frames intact and ~ccl_kernel_lifecycle_guard
            // never submits the exit command that releases the resident work-groups.
            try {
                process_host_initiated_op(host_op, m_producer, m_lifecycle, m_staging, m_queue);
            }
            catch (const std::exception& e) {
                CCL_FATAL("[proxy] host-initiated op failed: ", e.what());
            }
            backoff.reset();
            continue;
        }

        // poll device-initiated operations (FIFO cursor over the request ring)
        try {
            if (poll_device_op()) {
                backoff.reset();
                continue;
            }
        }
        catch (const std::exception& e) {
            CCL_FATAL("[proxy] device-initiated request failed: ", e.what());
        }

        // No work available; a queued op whose kernel has not started yet is not idle
        if (host_op_pending)
            backoff.reset();
        backoff.idle();
    }

    // Last action: tells stop() that join() cannot block
    m_exited.store(true, std::memory_order_release);
}

/* ======== Proxy service ======== */

ccl_proxy_service::ccl_proxy_service() = default;
ccl_proxy_service::~ccl_proxy_service() {
    // Runs from global_data::reset() at finalize. stop() frees device resources
    // through ZE_CALL, which throws on failure (e.g. device lost); escaping this
    // implicitly noexcept destructor would call std::terminate().
    // A failed teardown is only worth reporting: whatever it skipped is leaked.
    try {
        stop();
    }
    catch (const std::exception& e) {
        LOG_ERROR("[proxy] teardown failed: ", e.what());
    }
}

void ccl_proxy_service::init(sycl::queue& q) {
    // Initialize the proxy algorithm dispatch table
    // should be once per process
    init_proxy_algo_table();

    // Create an auxiliary in-order queue on the same device
    // for proxy operations and staging buffer management
    m_proxy_sycl_queue =
        new sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});

    m_cmd_ctx.init(*m_proxy_sycl_queue);
    m_req_ctx.init(*m_proxy_sycl_queue);
    m_staging.init(*m_proxy_sycl_queue);
}

void ccl_proxy_service::start() {
    if (m_started)
        return;

    m_proxy_thread = std::make_unique<ccl_proxy_thread>();
    m_proxy_thread->start(&m_cmd_ctx.producer(),
                          m_cmd_ctx.lifecycle(),
                          &m_staging,
                          m_proxy_sycl_queue,
                          &m_req_ctx.consumer());
    m_started = true;
}

void ccl_proxy_service::stop() {
    if (!m_started)
        return;

    if (m_proxy_thread) {
        if (!m_proxy_thread->stop()) {
            // The thread was detached and still dereferences its producer, request consumer,
            // staging, queue and its own members — so nothing here may be freed: not the
            // ring/completions/lifecycle, not the request ring, not the staging context, not the
            // queue, not the ccl_proxy_thread itself (release(), not reset()). A deliberate
            // shutdown leak; the alternative is a use-after-free, and a device-side one needs a
            // GPU reset to clear.
            m_proxy_thread.release();
            m_started = false;
            return;
        }
        m_proxy_thread.reset();
    }
    // A joined proxy thread means no *host* side still touches the ring.
    // It is not proof that the device side is done.
    m_cmd_ctx.destroy(*m_proxy_sycl_queue);
    m_req_ctx.destroy(*m_proxy_sycl_queue);
    m_staging.destroy(*m_proxy_sycl_queue);

    delete m_proxy_sycl_queue;
    m_proxy_sycl_queue = nullptr;

    m_started = false;
}

void ccl_proxy_service::reset() {
    m_cmd_ctx.reset();
    m_req_ctx.reset();
}

void ccl_proxy_service::submit(const ccl_proxy_op& op) {
    if (m_proxy_thread)
        m_proxy_thread->submit(op);
}
