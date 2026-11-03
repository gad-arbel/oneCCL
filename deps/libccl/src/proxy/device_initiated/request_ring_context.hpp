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

/* request_ring_context.hpp
 *
 * Owns the allocations behind the device-initiated request ring.
 *
 * Mirror of cmd_ring_context.hpp with the two allocations swapped, so that in
 * both directions the consumer polls memory local to it:
 *   - the ring is host-pinned USM, polled by the proxy thread;
 *   - the completions array is device memory, polled by the kernel, plus a
 *     GDR-backed host mapping so the proxy thread can write it.
 */

#pragma once

#include "proxy/device_initiated/request_ring_types.hpp"
#include "proxy/device_initiated/request_ring_host.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <sycl/sycl.hpp>
#include <cstring>
#include <stdexcept>

class ccl_request_ring_context {
public:
    ccl_request_ring_context() = default;
    // Note: destroy() must be called explicitly with the queue before destructor
    ~ccl_request_ring_context() = default;

    ccl_request_ring_context(const ccl_request_ring_context&) = delete;
    ccl_request_ring_context& operator=(const ccl_request_ring_context&) = delete;

    void init(sycl::queue& q) {
        // Request ring: host-pinned memory for optimal host polling
        m_ring = static_cast<ccl_request_ring*>(
            sycl::aligned_alloc_host(CACHELINE_SIZE, sizeof(ccl_request_ring), q));
        if (!m_ring) {
            throw std::runtime_error("Failed to allocate ccl_request_ring in host-pinned memory");
        }
        ccl_request_ring::init_slots(m_ring);

        // Completions: device memory for optimal device polling + GDR mmap for host access
        m_completions = static_cast<ccl_request_completions*>(
            sycl::aligned_alloc_device(CACHELINE_SIZE, sizeof(ccl_request_completions), q));
        if (!m_completions) {
            sycl::free(m_ring, q);
            m_ring = nullptr;
            throw std::runtime_error("Failed to allocate ccl_request_completions in device memory");
        }

        /* Seeded by a device-side copy, before the GDR mapping exists rather than through it, as
         * ccl_cmd_ring_context::init() does: the mapping is built for the consumer's complete()
         * MOVDIR64B alone, so correct initialization does not depend on it. Not cosmetic - a
         * kernel reading an unseeded next_slot claims an arbitrary sequence the host's cursor
         * never matches, wedging the ring with no request ever served. */
        ccl_request_completions seed;
        ccl_request_completions::init_completions(&seed);
        q.memcpy(m_completions, &seed, sizeof(ccl_request_completions)).wait();

        // Create GDR-backed host mapping for the device completions array
        try {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
            m_completions_host = static_cast<ccl_request_completions*>(
                ccl::ze::ze_ipc_mmap_device_allocation(ze_ctx,
                                                       m_completions,
                                                       sizeof(ccl_request_completions),
                                                       &m_map_base,
                                                       &m_map_size));
        }
        catch (const std::exception& e) {
            sycl::free(m_completions, q);
            m_completions = nullptr;
            sycl::free(m_ring, q);
            m_ring = nullptr;
            throw std::runtime_error(std::string("Failed to mmap device completions: ") + e.what());
        }

        m_consumer.init(m_ring, m_completions_host);
    }

    // NOTE: intentionally empty, as ccl_cmd_ring_context::reset() is - the sequences, completion
    // words and next_slot are monotonic wrap-safe counters, never reset between ops.
    void reset() {
        // Deliberately empty. Kept for API symmetry with ccl_cmd_ring_context.
    }

    void destroy(sycl::queue& q) {
        // Unmap the GDR mapping before freeing device memory. Unmapped by its own base,
        // not by m_completions_host, which may sit at an offset inside it.
        if (m_map_base) {
            ccl::ze::ze_ipc_munmap_device_allocation(m_map_base, m_map_size);
            m_map_base = nullptr;
            m_map_size = 0;
            m_completions_host = nullptr;
        }

        if (m_completions) {
            sycl::free(m_completions, q);
            m_completions = nullptr;
        }

        if (m_ring) {
            sycl::free(m_ring, q);
            m_ring = nullptr;
        }
    }

    ccl_request_ring* ring() const {
        return m_ring;
    }
    ccl_request_completions* completions() const {
        return m_completions;
    }
    ccl_request_ring_consumer& consumer() {
        return m_consumer;
    }

    // POD handed to kernels: both pointers as the device sees them.
    ccl_device_ring_handle handle() const {
        return ccl_device_ring_handle{ m_ring, m_completions };
    }

private:
    ccl_request_ring* m_ring = nullptr; // host-pinned ring (device writes it directly)
    ccl_request_completions* m_completions = nullptr; // device memory completions
    ccl_request_completions* m_completions_host = nullptr; // GDR-backed host-mapped alias
    void* m_map_base = nullptr; // base of that mapping, for munmap
    size_t m_map_size = 0;
    ccl_request_ring_consumer m_consumer;
};
