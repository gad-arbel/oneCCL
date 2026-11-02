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

#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_large_sycl_impl.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/selection/policy/policy_env.hpp"

ccl::event allgatherv_large(sycl::queue &q,
                            const void *send_buf,
                            size_t send_count,
                            void *recv_buf,
                            const ccl::vector_class<size_t> &recv_counts,
                            const ccl::vector_class<size_t> &offsets,
                            ccl::datatype dtype,
                            ccl_comm *comm,
                            const std::shared_ptr<ccl_comm> p2p_comm,
                            ccl_stream *global_stream,
                            const ccl::vector_class<ccl::event> &deps,
                            sycl_scaleup_options coll_attr) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("invoking MT allgatherv_large");
        ccl::global_data::env().sycl_allgatherv_tmp_buf = 1;
        CCL_THROW_IF_NOT(ccl::global_data::env().sycl_allgatherv_tmp_buf == 1,
                         "MT large kernel doesnt support disabled tmp buf");
    }
    else {
        LOG_DEBUG("invoking allgatherv_large");
    }
    sycl_ptrs_type sycl_ptrs;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    const size_t comm_size = p2p_comm->size();

    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
    static const simple_chunk_size_table_t simple_chunk_sizes =
        parse_simple_chunk_size_spec(CCL_SYCL_ALLGATHERV_SIMPLE_CHUNK_SIZE,
                                     ccl::global_data::env().sycl_allgatherv_simple_chunk_size);
    // allgatherv with different send_count on different ranks does not reach here
    // the policy knob `simple_chunk_size`, when set for this call, replaces the spec
    size_t chunk_size = ccl::selection::call_env().sycl_allgatherv_simple_chunk_size_override
                            ? ccl::selection::call_env().sycl_allgatherv_simple_chunk_size_override
                            : select_simple_chunk_size(simple_chunk_sizes, send_count * dsize);
    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, send_count, dsize);
    // TODO : generalize constraints for different hardware.
    // kernels with remote access is best performant at 64 bytes alignment (sycl_kernels_line_size/2) on PVC
    const size_t align_size = ccl::global_data::env().sycl_kernels_line_size / 2;
    const bool is_aligned = (send_count * dsize) % align_size == 0;

    bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));

    // use tmp buf for types < 4 byte size with odd count or non 4 byte aligned data
    // use tmp buf when data count bytes is not 64 byte aligned
    // since tmp buf version performs better in that case
    bool is_tmp_used = ccl::global_data::env().sycl_allgatherv_tmp_buf ||
                       ((!use_full_vector || !is_aligned) && ccl::global_data::env().sycl_auto_use_tmp_buf);
    // scale-out step can force to use "tmp buffer version" for async support purposes
    is_tmp_used |= (coll_attr.force_use_tmp && ccl::global_data::env().sycl_force_use_tmp_buf_scaleout);
    is_tmp_used |= !comm->get_topo_manager().has_p2p_access();

    if (is_tmp_used) {
        CCL_THROW_IF_NOT(!ccl::selection::call_env().sycl_copy_engine || is_arc,
                         "allgatherv using copy engines not supported with tmp buffer");

        // global rank of pair_comm neighbors should be adjacent for using tmp buffer
        // i.e. the ranks can be 2,3 or 3,2 but not 1,3
        CCL_ASSERT(pair_comm->size() <= 2);
        if (pair_comm->size() == 2) {
            const int rank_diff = pair_comm->get_node_rank(0) - pair_comm->get_node_rank(1);
            CCL_THROW_IF_NOT(
                abs(rank_diff) == 1,
                "communicator rank reordering not allowed with tmp buffer, set CCL_SYCL_ALLGATHERV_TMP_BUF=0 and CCL_SYCL_AUTO_USE_TMP_BUF=0");
        }
    }

    if (!is_tmp_used) {
        // synchronize SYCL host tasks, which can contain MPI calls,
        // with an MPI call (allgather) in ipc exchange
        if (coll_attr.wait_on_deps) {
            std::vector<sycl::event> dep_events = get_sycl_events(deps);
            for (auto &dep : dep_events) {
                dep.wait();
            }
        }
    }

    if (is_arc) {
        if (!is_tmp_used) {
            // only need output buffer
            std::vector<void *> registered_recv_ptrs =
                node_comm->get_registered_ptrs(recv_buf, send_count * comm_size * dsize);
            if (registered_recv_ptrs.size()) {
                LOG_DEBUG("allgatherv pointers are registered \n");
                sycl_ptrs.node_ptrs_wr =
                    get_ipc_ptrs<void, MAX_NODE_RANKS>(registered_recv_ptrs, comm, p2p_comm, (void *)recv_buf);
            }
            else {
                LOG_DEBUG("arc flow without registerd pointers filling node ptrs wr");
                std::vector<void *> ptrs{ recv_buf }; // index 0
                auto [sched, exchange_entry] = do_ipc_exchange(p2p_comm.get(), global_stream, ptrs);

                sycl_ptrs.node_ptrs_wr = get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 0, recv_buf, sched);

                delete exchange_entry;
                delete sched;
            }
        }
    }
    else { // PVC algorithms
        // if tmp buffer is not used, perform ipc exchange on send and recv buffer
        if (!is_tmp_used) {
            ccl_sched *sched = NULL;
            ze_handle_exchange_entry *exchange_entry = NULL;
            std::vector<void *> registered_send_ptrs =
                node_comm->get_registered_ptrs(send_buf, send_count * dsize);
            std::vector<void *> registered_recv_ptrs =
                node_comm->get_registered_ptrs(recv_buf, send_count * comm_size * dsize);
            if (registered_recv_ptrs.size() && registered_send_ptrs.size()) {
                LOG_DEBUG("allgather pointers are registered \n");
                sycl_ptrs.xelink_ptrs_rd =
                    get_ipc_ptrs<void, MAX_NODE_RANKS>(registered_send_ptrs, comm, even_comm, (void *)send_buf);
                sycl_ptrs.xelink_ptrs_wr =
                    get_ipc_ptrs<void, MAX_NODE_RANKS>(registered_recv_ptrs, comm, even_comm, (void *)recv_buf);
            }
            else {
                LOG_DEBUG("non arc flow without registerd pointers filling xelink ptrs rd/wr");
                std::vector<void *> ptrs{ (void *)send_buf, recv_buf }; // index 0 and 1
                auto p = do_ipc_exchange(comm, global_stream, ptrs);
                sched = p.first;
                exchange_entry = p.second;

                sycl_ptrs.xelink_ptrs_rd = get_ipc_ptrs<void, MAX_NODE_RANKS>(even_comm, 0, (void *)send_buf, sched);
                sycl_ptrs.xelink_ptrs_wr = get_ipc_ptrs<void, MAX_NODE_RANKS>(even_comm, 1, recv_buf, sched);
            }
            // use full vector (>= 8 bytes) if remote buffers and data size are 4 byte aligned
            use_full_vector =
                use_full_vector &&
                all_aligned(sycl_ptrs.xelink_ptrs_rd.data(), even_comm->size(), send_count, dsize, 4) &&
                all_aligned(sycl_ptrs.xelink_ptrs_wr.data(), even_comm->size(), send_count, dsize, 4);

            if (pair_comm->size() > 1) {
                LOG_DEBUG("pair comm > 1, filling mdfi ptrs rd/wr");
                assert(pair_comm->size() == MAX_TILES);
                int peer_pair_rank = pair_comm->rank() ? 0 : 1;
                if (registered_recv_ptrs.size() && registered_send_ptrs.size()) {
                    sycl_ptrs.mdfi_ptr_rd = get_ipc_ptrs<void, MAX_TILES>(
                        registered_send_ptrs, comm, pair_comm, (void *)send_buf)[peer_pair_rank];
                    sycl_ptrs.mdfi_ptr_wr = get_ipc_ptrs<void, MAX_TILES>(
                        registered_recv_ptrs, comm, pair_comm, (void *)recv_buf)[peer_pair_rank];
                }
                else {
                    sycl_ptrs.mdfi_ptr_rd =
                        get_ipc_ptrs<void, MAX_TILES>(pair_comm, 0, (void *)send_buf, sched)[peer_pair_rank];
                    sycl_ptrs.mdfi_ptr_wr =
                        get_ipc_ptrs<void, MAX_TILES>(pair_comm, 1, recv_buf, sched)[peer_pair_rank];
                }
                use_full_vector = use_full_vector &&
                                  all_aligned(&sycl_ptrs.mdfi_ptr_rd, 1, send_count, dsize, 4) &&
                                  all_aligned(&sycl_ptrs.mdfi_ptr_wr, 1, send_count, dsize, 4);
            }
            delete exchange_entry;
            delete sched;
        }
        else {
            sycl_ptrs.xelink_ptrs_rd = get_remote_large_tmp_buf(0, even_comm.get());
            if (pair_comm->size() > 1) {
                assert(pair_comm->size() == MAX_TILES);
                int peer_pair_rank = pair_comm->rank() ? 0 : 1;
                sycl_ptrs.mdfi_ptr_wr = get_remote_large_tmp_buf(0, pair_comm.get())[peer_pair_rank];
            }
        }
    } // if is_arc

    // the L0 copy engine signals peers with one flag that cannot tell chunks apart, so a chunk can
    // lose the next chunk's signal and hang; a copy engine gains nothing from chunking anyway.
    // allgatherv_large_impl() picks the copy engine per call, so check the variable enabling it.
    if (ccl::global_data::env().enable_ze_ce && chunk_size) {
        LOG_WARN_ONCE("allgatherv: CCL_ENABLE_ZE_CE=1 runs the large allgatherv unchunked, ",
                      "ignoring chunk size ",
                      chunk_size,
                      "; ");
        chunk_size = 0;
    }

    return for_each_chunk(chunk_size, send_count, dsize, [&](size_t offset, size_t pack_count) {
        const size_t offset_bytes = offset * dsize;
        // only peer send buffers are shifted; recv blocks move through the chunk offsets
        sycl_ptrs_type chunk_ptrs = is_tmp_used ? sycl_ptrs : offset_sycl_ptrs(sycl_ptrs, offset_bytes, 0);
        const void *chunk_send_buf = (const char *)send_buf + offset_bytes;
        ccl::vector_class<size_t> chunk_counts(recv_counts);
        ccl::vector_class<size_t> chunk_offsets(offsets);
        if (pack_count != send_count) {
            // kernels index offsets by node rank, so cover the whole node
            const size_t offsets_size =
                offsets.empty() ? std::max(comm_size, (size_t)node_comm->size()) : offsets.size();
            chunk_counts.assign(recv_counts.size(), pack_count);
            chunk_offsets.resize(offsets_size);
            for (size_t r = 0; r < offsets_size; r++) {
                chunk_offsets[r] = (offsets.empty() ? r * send_count * dsize : offsets[r]) + offset_bytes;
            }
        }
        auto lambda = [&]<typename T>() {
            if (use_full_vector) {
                return allgatherv_large_impl<T, true>(q,
                                                      chunk_send_buf,
                                                      pack_count,
                                                      send_count,
                                                      recv_buf,
                                                      chunk_counts,
                                                      chunk_offsets,
                                                      dtype,
                                                      comm,
                                                      p2p_comm,
                                                      global_stream,
                                                      chunk_ptrs,
                                                      deps,
                                                      is_tmp_used);
            }
            else {
                return allgatherv_large_impl<T, false>(q,
                                                       chunk_send_buf,
                                                       pack_count,
                                                       send_count,
                                                       recv_buf,
                                                       chunk_counts,
                                                       chunk_offsets,
                                                       dtype,
                                                       comm,
                                                       p2p_comm,
                                                       global_stream,
                                                       chunk_ptrs,
                                                       deps,
                                                       is_tmp_used);
            }
        };
        return invoke_collective(lambda, dtype);
    });
}
