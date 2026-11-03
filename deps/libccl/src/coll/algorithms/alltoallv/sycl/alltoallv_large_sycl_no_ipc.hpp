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

#pragma once
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/selection/policy/policy_env.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"

// peer_comm is the set of ranks the data is exchanged within (the p2p domain for
// alltoall, the node comm for alltoallv). The counts/displs arrays are indexed
// by peer_comm rank and must hold peer_comm->size() entries.
template <typename T>
sycl::event alltoallv_large_no_ipc_tmp_buf(sycl::queue& q,
                                           const void* send_buf,
                                           const size_t* send_counts,
                                           const size_t* send_displs,
                                           void* recv_buf,
                                           const size_t* recv_counts,
                                           const size_t* recv_displs,
                                           size_t max_count,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           const std::shared_ptr<ccl_comm> peer_comm,
                                           ccl_stream* global_stream,
                                           const ccl::vector_class<ccl::event>& deps) {
    const int world = peer_comm->size();
    const int rank = peer_comm->rank();

    // remote_tmp below is indexed by peer_comm rank, so the comm cannot be wider
    CCL_THROW_IF_NOT(world <= (int)MAX_NODE_RANKS,
                     "alltoall simple: peer comm size ",
                     world,
                     " exceeds MAX_NODE_RANKS ",
                     MAX_NODE_RANKS);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    // auto keeps the copy engine
    const bool use_memcpy =
        ccl::selection::call_env().sycl_alltoall_copy_engine_mode != ccl::copy_engine_kernel;

    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, max_count, dsize);
    for (int i = 0; i < world && use_full_vector; i++) {
        use_full_vector = send_displs[i] % 4 == 0 && recv_displs[i] % 4 == 0 &&
                          (send_counts[i] * dsize) % 4 == 0 && (recv_counts[i] * dsize) % 4 == 0;
    }
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 &&
           work_group_size <= q.get_device().get_info<sycl::info::device::max_work_group_size>());

    constexpr int num_stages = 2;
    const size_t pool_bytes =
        get_tmp_buf_size_per_rank() * ccl::global_data::get().get_local_proc_count();
    const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
    const size_t slot_bytes = pool_bytes / (num_stages * world) / align_bytes * align_bytes;
    const size_t max_chunk_count = slot_bytes / dsize;
    CCL_THROW_IF_NOT(
        max_chunk_count > 0, "alltoall simple: tmp buffer too small for ", world, " ranks");

    LOG_DEBUG("alltoall simple (no IPC) kernel rank: ",
              rank,
              " comm size: ",
              world,
              " max_count:",
              max_count,
              " slot_bytes:",
              slot_bytes,
              " max_chunk_count:",
              max_chunk_count);

    char* local_tmp = (char*)get_large_tmp_buf(0, comm);
    std::array<void*, MAX_NODE_RANKS> remote_tmp = get_remote_large_tmp_buf(0, peer_comm.get());

    auto submit_copy = [&](void* dst, const void* src, size_t chunk_count) {
        if (use_memcpy) {
            q.memcpy(dst, src, chunk_count * dsize);
        }
        else {
            const size_t kernel_threads = chunk_count / vec_size + chunk_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
            sycl::nd_range<1> nd_range(kernel_size, work_group_size);
            void* src_nc = const_cast<void*>(src);
            q.parallel_for(nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                copy_kernel_wrap<T, full_vec_size, partial_vec_size>(
                    dst, src_nc, chunk_count, use_full_vector);
            });
        }
    };

    // honour explicit deps without paying for a barrier: the queue is in-order, so
    // a single wait submission orders everything that follows behind them.
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (!dep_events.empty()) {
        submit_wait_on_events(q, dep_events);
    }

    // the block this rank sends to itself stays on the device
    if (send_counts[rank] > 0) {
        submit_copy((char*)recv_buf + recv_displs[rank],
                    (const char*)send_buf + send_displs[rank],
                    send_counts[rank]);
    }

    int stage = 0;
    size_t offset = 0;
    while (offset < max_count) {
        const size_t chunk_count = std::min(max_chunk_count, max_count - offset);
        const size_t stage_base = (size_t)stage * world;

        // scatter: this rank's block for each peer -> peer's tmp slot[stage][rank].
        // A peer whose count is already exhausted just contributes nothing to this
        // chunk; the barriers sit outside this loop so skipping stays in lockstep.
        for (int i = 1; i < world; i++) {
            const int peer = (rank + i) % world;
            if (send_counts[peer] <= offset) {
                continue;
            }
            const size_t n = std::min(chunk_count, send_counts[peer] - offset);
            const char* src = (const char*)send_buf + send_displs[peer] + offset * dsize;
            char* dst = (char*)remote_tmp[peer] + (stage_base + rank) * slot_bytes;
            submit_copy(dst, src, n);
        }
        // full all-ranks barrier, not a neighbour handshake - see the note above
        invoke_barrier(peer_comm, q, {}, is_cpu_barrier);

        // gather: own tmp slot[stage][p] holds the block that rank p sent here.
        // recv_counts[p] equals rank p's send_counts[rank], so the length matches
        // what p actually wrote.
        for (int i = 1; i < world; i++) {
            const int p = (rank + i) % world;
            if (recv_counts[p] <= offset) {
                continue;
            }
            const size_t n = std::min(chunk_count, recv_counts[p] - offset);
            const char* src = local_tmp + (stage_base + p) * slot_bytes;
            char* dst = (char*)recv_buf + recv_displs[p] + offset * dsize;
            submit_copy(dst, src, n);
        }

        stage ^= 1;
        offset += chunk_count;
    }

    return invoke_barrier(peer_comm, q, {}, is_cpu_barrier);
}
