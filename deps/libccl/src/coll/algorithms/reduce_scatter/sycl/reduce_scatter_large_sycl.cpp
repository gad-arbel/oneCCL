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

#include "coll/algorithms/utils/sycl_custom_preop.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_impl.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"
#include "coll/selection/policy/policy_env.hpp"

ccl::event reduce_scatter_large(const void *send_buf,
                                void *recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                std::shared_ptr<ccl_comm> p2p_comm,
                                ccl_stream *global_stream,
                                const ccl::vector_class<ccl::event> &deps,
                                sycl_scaleup_options coll_attr) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("invoking MT reduce_scatter_large");
        ccl::global_data::env().sycl_reduce_scatter_tmp_buf = 1;
        CCL_THROW_IF_NOT(ccl::global_data::env().sycl_reduce_scatter_tmp_buf == 1,
                         "MT large kernel doesnt support disabled tmp buf");
    }
    else {
        LOG_DEBUG("invoking reduce_scatter_large");
    }

    sycl_ptrs_type sycl_ptrs;
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
    static const simple_chunk_size_table_t simple_chunk_sizes =
        parse_simple_chunk_size_spec(CCL_SYCL_REDUCE_SCATTER_SIMPLE_CHUNK_SIZE,
                                     ccl::global_data::env().sycl_reduce_scatter_simple_chunk_size);
    // the policy knob `simple_chunk_size`, when set for this call, replaces the spec
    const size_t chunk_size =
        ccl::selection::call_env().sycl_reduce_scatter_simple_chunk_size_override
            ? ccl::selection::call_env().sycl_reduce_scatter_simple_chunk_size_override
            : select_simple_chunk_size(simple_chunk_sizes, recv_count * p2p_comm->size() * dsize);

    // BMG path
    if (is_arc_card(ccl::global_data::get().ze_data->devices[0].family)) {
        // tmp_buf flag ignored, since temp buffer is always used,
        // there is no algorithm which does not use tmp buf
        bool is_tmp_used = ccl::global_data::env().sycl_reduce_scatter_tmp_buf;
        is_tmp_used |= !node_comm->get_topo_manager().has_p2p_access();
        // The BMG su_ring path uses reduce_pair_dispatch for the in-flight reduction, which does not apply the
        // per-rank scalar multiply.
        // For custom (pre_mul_sum) reductions, stage the full input (recv_count * world) into a scratch buffer
        // with the scalar applied, then run the ring with SUM transport semantics over the preprocessed staging buffer.
        // recv_buf cannot be reused because it only holds this rank's slice (recv_count).
        const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);
        const size_t world_count = recv_count * p2p_comm->size();
        const size_t world_bytes =
            world_count * ccl::global_data::get().dtypes->get(dtype).size();
        // The scratch is fixed-size; if the full input doesn't fit, skip the BMG branch and fall
        // through to reduce_scatter_large_impl which handles pre-op natively (copy_and_modify_data)
        // and chunks internally. Chunking inside this branch would duplicate the multi-node
        // dispatcher's rearrange/chunk logic for a regime that's not a known hot path.
        const bool staging_fits =
            world_bytes <= (size_t)comm->get_scaleout_device_buf_size();
        LOG_DEBUG("invoking reduce_scatter_large BMG path: reduce_has_pre_operation: ", reduce_has_pre_operation, " staging_fits: ", staging_fits);
        if (!reduce_has_pre_operation || staging_fits) {
            const void *transport_send_buf = send_buf;
            ccl::reduction transport_reduction = reduction;
            ccl::vector_class<ccl::event> transport_deps_local;
            void *preop_staging_buf = nullptr;
            if (reduce_has_pre_operation) {
                sycl::queue q = global_stream->get_native_stream();
                preop_staging_buf = comm->get_scaleout_device_buf(q);
                ccl::event preop_event = stage_and_apply_custom_preop(
                    q, preop_staging_buf, send_buf, world_count, dtype, reduction,
                    p2p_comm->get_tmp_buf_size(), get_sycl_events(deps));
                transport_send_buf = preop_staging_buf;
                transport_reduction = ccl::reduction::sum;
                transport_deps_local.push_back(std::move(preop_event));
            }
            const ccl::vector_class<ccl::event> &transport_deps =
                reduce_has_pre_operation ? transport_deps_local : deps;
            // chunks are column slices of every rank block, so the ring reads a recv_count stride
            ccl::event e = for_each_chunk(chunk_size, recv_count, dsize, [&](size_t offset, size_t pack_count) {
                auto lambda = [&]<typename T>() {
                    return reduce_scatter_large_su_ring<T>(
                        (const char *)transport_send_buf + offset * dsize, (char *)recv_buf + offset * dsize,
                        pack_count, recv_count, dtype, transport_reduction, comm, p2p_comm, global_stream,
                        sycl_ptrs, transport_deps);
                };
                sycl::event e = invoke_collective_sycl(lambda, dtype);
                return ccl::event::create_from_native(e);
            });
            if (preop_staging_buf) {
                comm->put_scaleout_device_buf(preop_staging_buf);
            }
            return e;
        }
        // else: fall through to the non-BMG path below.
    }

    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, recv_count, dsize);
    // TODO : generalize constraints for different hardware.
    // kernels with remote access is best performant at 64 bytes alignment (sycl_kernels_line_size/2) on PVC
    const size_t align_size = ccl::global_data::env().sycl_kernels_line_size / 2;
    const bool is_aligned = (recv_count * dsize) % align_size == 0;
    // use tmp buf for types < 4 byte size with odd count or non 4 byte aligned data
    // use tmp buf when data count bytes is not 64 byte aligned
    // since tmp buf version performs better in that case
    bool is_tmp_used = ccl::global_data::env().sycl_reduce_scatter_tmp_buf ||
                       ((!use_full_vector || !is_aligned) && ccl::global_data::env().sycl_auto_use_tmp_buf);
    if (even_comm->size() == 1) {
        is_tmp_used = ccl::global_data::env().sycl_reduce_scatter_tmp_buf;
    }
    // scale-out step can force to use "tmp buffer version" for async support purposes
    is_tmp_used |= (coll_attr.force_use_tmp && ccl::global_data::env().sycl_force_use_tmp_buf_scaleout);
    // user defined reduction operations require to apply special scalar before
    // doing reduction itself. Since user buffer (send_buf) cannot be modified,
    // a temporary buffer is needed to hold the modified data.
    is_tmp_used |= ccl_reduction_type_storage::is_custom(reduction);

    if (is_tmp_used) {
        // global rank of pair_comm neighbors should be adjacent for using tmp buffer
        // i.e. the ranks can be 2,3 or 3,2 but not 1,3
        CCL_ASSERT(pair_comm->size() <= 2);
        if (pair_comm->size() == 2) {
            const int rank_diff = pair_comm->get_node_rank(0) - pair_comm->get_node_rank(1);
            CCL_THROW_IF_NOT(
                abs(rank_diff) == 1,
                "communicator rank reordering not allowed with tmp buffer, set CCL_SYCL_REDUCE_SCATTER_TMP_BUF=0 and CCL_SYCL_AUTO_USE_TMP_BUF=0");
        }
    }

    if (!is_tmp_used) {
        ccl_sched *sched = NULL;
        ze_handle_exchange_entry *exchange_entry = NULL;
        const size_t comm_size = p2p_comm->size();

        std::vector<void *> registered_send_ptrs =
            node_comm->get_registered_ptrs(send_buf, recv_count * dsize * comm_size);

        if (registered_send_ptrs.size()) {
            LOG_DEBUG("reduce_scatter pointers are registered \n");
            sycl_ptrs.xelink_ptrs_rd =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(registered_send_ptrs, p2p_comm.get(), even_comm, (void *)send_buf);
        }
        else {
            std::vector<void *> ptrs{ (void *)send_buf, recv_buf }; // index 0 and 1
            auto p = do_ipc_exchange(comm, global_stream, ptrs);
            sched = p.first;
            exchange_entry = p.second;

            sycl_ptrs.xelink_ptrs_rd = get_ipc_ptrs<void, MAX_NODE_RANKS>(even_comm, 0, (void *)send_buf, sched);
        }

        // use full vector (>= 8 bytes) if remote buffers and data size are 4 byte aligned
        use_full_vector = use_full_vector &&
                          all_aligned(sycl_ptrs.xelink_ptrs_rd.data(), even_comm->size(), recv_count, dsize, 4);

        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            if (registered_send_ptrs.size()) {
                sycl_ptrs.mdfi_ptr_rd = get_ipc_ptrs<void, MAX_TILES>(
                    registered_send_ptrs, comm, pair_comm, (void *)send_buf)[peer_pair_rank];
            }
            else {
                sycl_ptrs.mdfi_ptr_rd =
                    get_ipc_ptrs<void, MAX_TILES>(pair_comm, 0, (void *)send_buf, sched)[peer_pair_rank];
            }
            use_full_vector = use_full_vector && all_aligned(&sycl_ptrs.mdfi_ptr_rd, 1, recv_count, dsize, 4);
        }
        delete exchange_entry;
        delete sched;
    }
    else {
        // 0 index is used for tmp work buffer and
        // 1 index is used to copy input data
        sycl_ptrs.xelink_ptrs_rd = get_remote_large_tmp_buf(1, even_comm.get());
        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            sycl_ptrs.mdfi_ptr_rd = get_remote_large_tmp_buf(1, pair_comm.get())[peer_pair_rank];
        }
    }

    return for_each_chunk(chunk_size, recv_count, dsize, [&](size_t offset, size_t pack_count) {
        const size_t offset_bytes = offset * dsize;
        // tmp-buffer peers are not user buffers, so they keep their base
        sycl_ptrs_type chunk_ptrs = is_tmp_used ? sycl_ptrs : offset_sycl_ptrs(sycl_ptrs, offset_bytes, 0);
        const void *chunk_send_buf = (const char *)send_buf + offset_bytes;
        void *chunk_recv_buf = (char *)recv_buf + offset_bytes;
        auto lambda = [&]<typename T>() {
            if (use_full_vector) {
                return reduce_scatter_large_impl<T, true>(chunk_send_buf,
                                                          chunk_recv_buf,
                                                          pack_count,
                                                          recv_count,
                                                          dtype,
                                                          reduction,
                                                          comm,
                                                          global_stream,
                                                          chunk_ptrs,
                                                          deps,
                                                          is_tmp_used);
            }
            else {
                return reduce_scatter_large_impl<T, false>(chunk_send_buf,
                                                           chunk_recv_buf,
                                                           pack_count,
                                                           recv_count,
                                                           dtype,
                                                           reduction,
                                                           comm,
                                                           global_stream,
                                                           chunk_ptrs,
                                                           deps,
                                                           is_tmp_used);
            }
        };
        return invoke_collective(lambda, dtype);
    });
}
