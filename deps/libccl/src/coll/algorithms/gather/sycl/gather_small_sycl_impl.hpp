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
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"

// gather_small, push mode: every rank writes its block into root's tmp buffer at
// slot rank * count, comm barrier, root copies the staged region into recv_buf.
// Root stages its own block too, which keeps the copy-out contiguous and makes
// the in-place case fall out for free.
//
// is_recording carries the SYCL-graph generation index. ccl_gather rejects
// recording at its entry today, so those branches are inert.

// Kernel-name tag: distinct template args must yield distinct kernel names.
template <typename T, int VS, bool use_block, bool use_global_barrier>
class oneccl_gather_small {};

// Vector path. GB=true folds barrier and root read-out into this parallel_for;
// GB=false does the staging write only (local_tmp and recv then unused).
// Caller guarantees count % vec_size == 0, which is what makes the AT* cast on
// dst_slot sound for every rank.
template <typename T, int vec_size, bool use_global_barrier>
inline void gather_small_inner(const void* in,
                               void* root_tmp,
                               const void* local_tmp,
                               void* recv,
                               ccl_kernel_barrier_data kernel_barrier_data,
                               ccl_comm_barrier_data comm_barrier_data,
                               size_t comm_size,
                               int comm_rank,
                               int root,
                               size_t count_per_peer,
                               sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_count = count_per_peer / vec_size;

    {
        const T* src_slot = static_cast<const T*>(in);
        T* dst_slot = static_cast<T*>(root_tmp) + static_cast<size_t>(comm_rank) * count_per_peer;
        if (idx < packed_count) {
            reinterpret_cast<AT*>(dst_slot)[idx] = reinterpret_cast<const AT*>(src_slot)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count_per_peer) {
                dst_slot[new_idx] = src_slot[new_idx];
            }
        }
    }

    if constexpr (!use_global_barrier) {
        return;
    }
    else {
        // comm_barrier does not order this rank's own work-items, so without this
        // a signalling one could outrun the sibling stores
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);

        comm_barrier(comm_barrier_data, it);

        // reset local barrier counter
        if (idx == 0) {
            kernel_barrier_data.reset_sync_data();
        }

        if (comm_rank != root) {
            return;
        }
        const T* staged = static_cast<const T*>(local_tmp);
        T* out = static_cast<T*>(recv);
        if (idx < packed_count) {
#pragma unroll
            for (size_t p = 0; p < comm_size; p++) {
                const size_t slot_off = p * count_per_peer;
                reinterpret_cast<AT*>(out + slot_off)[idx] =
                    reinterpret_cast<const AT*>(staged + slot_off)[idx];
            }
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count_per_peer) {
#pragma unroll
                for (size_t p = 0; p < comm_size; p++) {
                    const size_t slot_off = p * count_per_peer;
                    out[slot_off + new_idx] = staged[slot_off + new_idx];
                }
            }
        }
    }
}

// Byte fallback, used when any alignment check fails.
template <bool use_global_barrier>
inline void gather_small_inner_bytes(const uint8_t* in,
                                     uint8_t* root_tmp,
                                     const uint8_t* local_tmp,
                                     uint8_t* recv,
                                     ccl_kernel_barrier_data kernel_barrier_data,
                                     ccl_comm_barrier_data comm_barrier_data,
                                     size_t comm_size,
                                     int comm_rank,
                                     int root,
                                     size_t bytes_per_peer,
                                     sycl::nd_item<1> it) {
    const size_t byte_idx = it.get_global_linear_id();

    if (byte_idx < bytes_per_peer) {
        uint8_t* dst_slot = root_tmp + static_cast<size_t>(comm_rank) * bytes_per_peer;
        dst_slot[byte_idx] = in[byte_idx];
    }

    if constexpr (!use_global_barrier) {
        return;
    }
    else {
        // local barrier within gpu, see gather_small_inner
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);

        comm_barrier(comm_barrier_data, it);

        if (byte_idx == 0) {
            kernel_barrier_data.reset_sync_data();
        }

        if (comm_rank == root && byte_idx < bytes_per_peer) {
#pragma unroll
            for (size_t p = 0; p < comm_size; p++) {
                const size_t slot_off = p * bytes_per_peer;
                recv[slot_off + byte_idx] = local_tmp[slot_off + byte_idx];
            }
        }
    }
}

template <typename T>
ccl::event gather_small_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
                             int root,
                             ccl_comm* comm,
                             std::shared_ptr<ccl_comm> p2p_comm,
                             ccl_stream* global_stream,
                             const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();
    // under graph capture the kernel advances tmp_buf_idx on-device
    const bool is_recording = use_recording_path(q);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    const int comm_size = p2p_comm->size();
    const int comm_rank = p2p_comm->rank();

    const bool use_full_vector =
        can_use_full_vector(send_buf, comm_rank == root ? recv_buf : send_buf, count, dsize);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    // in-kernel barriers only synchronise if every thread is resident at once,
    // so kernel_size must stay <= hw_threads
    const size_t hw_threads = get_total_threads(q);
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads) {
        num_threads = hw_threads;
    }
    const size_t sec_0 = std::min(num_threads, hw_threads);

    auto [local_tmp_buf, remote_ptrs] =
        is_recording ? p2p_comm->get_all_tmp_bufs_gpu(true) : p2p_comm->get_all_tmp_bufs(true);
    int* tmp_buf_idx = p2p_comm->get_tmp_buf_idx();
    int* tmp_buf_secondary_idx = p2p_comm->get_tmp_buf_secondary_idx();

    // root's buffer alone holds all comm_size blocks; re-assert so a mis-set
    // threshold fails loudly instead of overrunning it
    const size_t tmp_buf_size = p2p_comm->get_tmp_buf_size();
    const size_t bytes_per_peer = count * dsize;
    const size_t staged_bytes = static_cast<size_t>(comm_size) * bytes_per_peer;

    CCL_THROW_IF_NOT(staged_bytes <= tmp_buf_size,
                     "gather_small: comm_size*count*dsize (",
                     staged_bytes,
                     ") exceeds tmp buf size (",
                     tmp_buf_size,
                     " B). Lower CCL_SYCL_GATHER_SMALL_THRESHOLD or grow CCL_SYCL_TMP_BUF_SIZE.");

    // IPC bases are page aligned, so only the user buffers and the slot offset
    // need checking: slot p starts p * count elements in
    constexpr int vec_size = get_num_elements<T, 8>();
    const bool slot_vec_aligned = (count % vec_size == 0);
    // non-root may pass recv_buf == nullptr, so gate only on what we touch
    const bool ptrs_aligned_T =
        is_aligned(send_buf, count, dsize, alignof(T)) &&
        (comm_rank != root || is_aligned(recv_buf, count, dsize, alignof(T)));

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;
    // MT: ranks share the process, bracket to serialise generation-index updates
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    // VS : vec_size, SGS : sub_group_size (also wg size), GB : use_global_barrier
    auto gather_invoke = [=, &q]<int VS, int SGS, bool GB>(std::vector<sycl::event>& sycl_deps) {
        constexpr int wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / VS + count % VS;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
        ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data(comm).inc_slot();
        const ccl_comm_barrier_data comm_barrier_data =
            GB ? p2p_comm->barrier_inc() : p2p_comm->barrier_data();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for<oneccl_gather_small<T, VS, true, GB>>(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    size_t base_offset_bytes = 0;
                    if (is_recording) {
                        base_offset_bytes = *tmp_buf_idx * tmp_buf_size;
                    }
                    void* root_tmp =
                        static_cast<uint8_t*>(remote_ptrs[root]) + base_offset_bytes;
                    const void* my_tmp =
                        static_cast<const uint8_t*>(local_tmp_buf) + base_offset_bytes;
                    gather_small_inner<T, VS, GB>(send_buf,
                                                  root_tmp,
                                                  my_tmp,
                                                  recv_buf,
                                                  kernel_barrier_data,
                                                  comm_barrier_data,
                                                  comm_size,
                                                  comm_rank,
                                                  root,
                                                  count,
                                                  it);
                    if (is_recording && it.get_global_linear_id() == 0) {
                        *tmp_buf_secondary_idx = *tmp_buf_idx;
                        *tmp_buf_idx = (*tmp_buf_idx + 1) % ccl_tmp_bufs::buf_count;
                    }
                });
        });
        return local_event;
    };

    // GB : use_global_barrier
    auto gather_invoke_bytes = [=, &q]<bool GB>(std::vector<sycl::event>& sycl_deps) {
        constexpr int wg_size = 32, sg_size = 32;
        const size_t kernel_size = ((bytes_per_peer + wg_size - 1) / wg_size) * wg_size;
        ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data(comm).inc_slot();
        const ccl_comm_barrier_data comm_barrier_data =
            GB ? p2p_comm->barrier_inc() : p2p_comm->barrier_data();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for<oneccl_gather_small<T, 1, false, GB>>(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    size_t base_offset_bytes = 0;
                    if (is_recording) {
                        base_offset_bytes = *tmp_buf_idx * tmp_buf_size;
                    }
                    uint8_t* root_tmp =
                        static_cast<uint8_t*>(remote_ptrs[root]) + base_offset_bytes;
                    const uint8_t* my_tmp =
                        static_cast<const uint8_t*>(local_tmp_buf) + base_offset_bytes;
                    gather_small_inner_bytes<GB>(static_cast<const uint8_t*>(send_buf),
                                                 root_tmp,
                                                 my_tmp,
                                                 static_cast<uint8_t*>(recv_buf),
                                                 kernel_barrier_data,
                                                 comm_barrier_data,
                                                 comm_size,
                                                 comm_rank,
                                                 root,
                                                 bytes_per_peer,
                                                 it);
                    if (is_recording && it.get_global_linear_id() == 0) {
                        *tmp_buf_secondary_idx = *tmp_buf_idx;
                        *tmp_buf_idx = (*tmp_buf_idx + 1) % ccl_tmp_bufs::buf_count;
                    }
                });
        });
        return local_event;
    };

    // Split orchestrator: staging write -> barrier -> root copy-out. VS==1 is
    // the byte path.
    auto gather_memcpy = [=, &q]<int VS>(std::vector<sycl::event> sycl_deps) {
        sycl::event local_event;
        if constexpr (VS != 1) {
            local_event = gather_invoke.template operator()<VS, 32, false>(sycl_deps);
        }
        else {
            local_event = gather_invoke_bytes.template operator()<false>(sycl_deps);
        }
        sycl_deps.clear();
        sycl_deps.push_back(std::move(local_event));

        sycl::event barrier_event = invoke_barrier(p2p_comm, q, sycl_deps, use_cpu_barrier);
        if (comm_rank != root) {
            return barrier_event;
        }

        sycl::event memcpy_event;
        if (is_recording) {
            memcpy_event = kernel_memcpy(q,
                                         local_tmp_buf,
                                         recv_buf,
                                         tmp_buf_secondary_idx,
                                         nullptr,
                                         static_cast<size_t>(comm_size) * count,
                                         dsize,
                                         tmp_buf_size,
                                         { barrier_event });
        }
        else {
            memcpy_event = q.submit([=](sycl::handler& h) {
                h.depends_on(barrier_event);
                h.memcpy(recv_buf, local_tmp_buf, staged_bytes);
            });
        }
        return memcpy_event;
    };

    // these need control back on the host between phases, so they rule out GB
    const bool single_kernel_ok =
        !use_cpu_barrier && !is_recording && !comm->is_multi_thread_instance();
    const bool vec_path_ok = ptrs_aligned_T && use_full_vector && slot_vec_aligned;
    const bool fits_hw = count <= sec_0;
    if (single_kernel_ok && fits_hw && vec_path_ok) {
        LOG_DEBUG("gather_small: single-kernel vec path, count: ",
                  count,
                  ", comm_size: ",
                  comm_size);
        kernel_event = gather_invoke.template operator()<vec_size, 32, true>(dep_events);
    }
    else if (single_kernel_ok && fits_hw && !vec_path_ok && bytes_per_peer <= sec_0) {
        LOG_DEBUG("gather_small: single-kernel byte path, count: ",
                  count,
                  ", comm_size: ",
                  comm_size);
        kernel_event = gather_invoke_bytes.template operator()<true>(dep_events);
    }
    else if (vec_path_ok) {
        LOG_DEBUG("gather_small: split vec path, count: ", count, ", comm_size: ", comm_size);
        kernel_event = gather_memcpy.template operator()<vec_size>(dep_events);
    }
    else {
        LOG_DEBUG("gather_small: split byte path, count: ", count, ", comm_size: ", comm_size);
        kernel_event = gather_memcpy.template operator()<1>(dep_events);
    }
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    return ccl::event::create_from_native(kernel_event);
}
