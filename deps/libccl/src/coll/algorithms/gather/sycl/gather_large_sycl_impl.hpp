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
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

// Write protocol: each rank pushes its slice into root's recv_buf. One remote
// write per rank, issued concurrently.
inline ccl::event gather_large_impl_ipc_write(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        int root,
                                        ccl_comm* comm,
                                        std::shared_ptr<ccl_comm> p2p_comm,
                                        sycl::queue q,
                                        sycl_ptrs_type& sycl_ptrs,
                                        const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("gather large kernel ipc write");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int comm_rank = p2p_comm->rank();
    const int comm_size = p2p_comm->size();

    void* root_recv_buf = sycl_ptrs.node_ptrs_wr[root];

    // in-order queue: only the first submission needs explicit deps
    invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    const bool inplace =
        ccl::is_gather_inplace(send_buf, recv_buf, count, dsize, comm_rank, root, comm_size);
    if (inplace) {
        LOG_DEBUG("gather large inplace at root, no copy");
    }
    else {
        LOG_DEBUG("gather large copy into root slot, count: ", count);
        const size_t my_offset_bytes = static_cast<size_t>(comm_rank) * count * dsize;
        void* my_slot = static_cast<uint8_t*>(root_recv_buf) + my_offset_bytes;
        q.memcpy(my_slot, send_buf, count * dsize);
    }

    // root now knows every peer has finished writing
    sycl::event ret_event = invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);

    return ccl::event::create_from_native(ret_event);
}

// Read protocol: root pulls each rank's block out of that rank's send_buf. All
// comm_size-1 reads issue from root's queue.
inline ccl::event gather_large_impl_ipc_read(const void* send_buf,
                                             void* recv_buf,
                                             size_t count,
                                             ccl::datatype dtype,
                                             int root,
                                             ccl_comm* comm,
                                             std::shared_ptr<ccl_comm> p2p_comm,
                                             sycl::queue q,
                                             sycl_ptrs_type& sycl_ptrs,
                                             const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("gather large kernel ipc read");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int comm_rank = p2p_comm->rank();
    const int comm_size = p2p_comm->size();

    invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    if (comm_rank == root) {
        const bool inplace =
            ccl::is_gather_inplace(send_buf, recv_buf, count, dsize, comm_rank, root, comm_size);
        for (int p = 0; p < comm_size; p++) {
            if (p == root) {
                if (inplace) {
                    LOG_DEBUG("gather large read inplace at root, no copy");
                    continue;
                }
                void* own_dst =
                    static_cast<uint8_t*>(recv_buf) + static_cast<size_t>(p) * count * dsize;
                q.memcpy(own_dst, send_buf, count * dsize);
                continue;
            }
            void* dst = static_cast<uint8_t*>(recv_buf) + static_cast<size_t>(p) * count * dsize;
            q.memcpy(dst, sycl_ptrs.node_ptrs_rd[p], count * dsize);
        }
    }

    // peers now know root has finished reading their send_buf
    sycl::event ret_event = invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);

    return ccl::event::create_from_native(ret_event);
}

// Tmp-buf staged variant, chunked. Per chunk: every rank writes its sub-block
// into its offset of root's remote tmp slot, barrier, root drains its own slot.
// Chunks alternate between two slots so a write of chunk k+1 cannot clobber the
// slot root is still reading chunk k from.
//
// Slot layout: [ rank 0 sub-block ][ rank 1 sub-block ]...[ rank N-1 sub-block ]
inline ccl::event gather_large_impl_tmp(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        int root,
                                        ccl_comm* comm,
                                        std::shared_ptr<ccl_comm> p2p_comm,
                                        sycl::queue q,
                                        const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("gather large kernel tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int comm_rank = p2p_comm->rank();
    const int comm_size = p2p_comm->size();
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    std::vector<void*> tmp_bufs = { get_large_tmp_buf(0, comm), get_large_tmp_buf(1, comm) };
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_0 =
        get_remote_large_tmp_buf(0, p2p_comm.get());
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_1 =
        get_remote_large_tmp_buf(1, p2p_comm.get());
    std::vector<void*> remote_root_tmp_bufs = { remote_tmp_bufs_0[root],
                                                remote_tmp_bufs_1[root] };

    // one sub-block per rank must fit the slot, kernel_mem_align preserved
    const size_t slot_bytes_full = (char*)tmp_bufs[1] - (char*)tmp_bufs[0];
    const size_t per_peer_slot_bytes_raw = slot_bytes_full / static_cast<size_t>(comm_size);
    const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
    const size_t per_peer_slot_bytes = (per_peer_slot_bytes_raw / align_bytes) * align_bytes;
    CCL_THROW_IF_NOT(per_peer_slot_bytes >= dsize,
                     "gather_large_tmp: per-peer slot (",
                     per_peer_slot_bytes,
                     " B) smaller than one element (dsize=",
                     dsize,
                     ")");
    const size_t chunk_count = per_peer_slot_bytes / dsize;
    const size_t num_chunks = count / chunk_count + (count % chunk_count != 0);

    // in-order queue: deps go on the first staging memcpy
    bool first_submission = true;

    for (size_t i = 0; i < num_chunks; i++) {
        size_t copy_count = chunk_count;
        if (i == num_chunks - 1 && count % chunk_count != 0) {
            copy_count = count % chunk_count;
        }
        const size_t copy_bytes = copy_count * dsize;
        const size_t chunk_offset_bytes = i * chunk_count * dsize;

        void* my_tmp = tmp_bufs[i % 2];
        void* root_tmp_remote = remote_root_tmp_bufs[i % 2];

        {
            void* dst = static_cast<uint8_t*>(root_tmp_remote) +
                        static_cast<size_t>(comm_rank) * per_peer_slot_bytes;
            const void* src = static_cast<const uint8_t*>(send_buf) + chunk_offset_bytes;
            if (first_submission) {
                q.memcpy(dst, src, copy_bytes, dep_events);
                first_submission = false;
            }
            else {
                q.memcpy(dst, src, copy_bytes);
            }
        }
        if (comm->is_multi_thread_instance()) {
            pthread_barrier_wait(
                ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        }
        invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);

        if (comm_rank == root) {
            for (int p = 0; p < comm_size; p++) {
                void* recv_dst = static_cast<uint8_t*>(recv_buf) +
                                 static_cast<size_t>(p) * count * dsize + chunk_offset_bytes;
                const void* tmp_src = static_cast<const uint8_t*>(my_tmp) +
                                      static_cast<size_t>(p) * per_peer_slot_bytes;
                q.memcpy(recv_dst, tmp_src, copy_bytes);
            }
        }
    }
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    // stops a later call reusing a slot before root's last read
    sycl::event ret_event = invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);
    return ccl::event::create_from_native(ret_event);
}

// Not templated on dtype: byte movement only, so a per-dtype dispatch would
// compile eleven identical copies.
inline ccl::event gather_large_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    ccl_comm* comm,
                                    std::shared_ptr<ccl_comm> p2p_comm,
                                    ccl_stream* global_stream,
                                    sycl_ptrs_type& sycl_ptrs,
                                    bool is_tmp_used,
                                    const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();

    if (!is_tmp_used) {
        if (ccl::global_data::env().sycl_gather_large_protocol ==
            ccl_sycl_gather_protocol::read) {
            return gather_large_impl_ipc_read(
                send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
        }
        return gather_large_impl_ipc_write(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
    }
    return gather_large_impl_tmp(
        send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, deps);
}
