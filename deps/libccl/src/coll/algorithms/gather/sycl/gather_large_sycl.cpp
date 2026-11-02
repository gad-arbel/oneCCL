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

// Resolves the IPC peer pointers and picks the data path; the paths themselves
// live in gather_large_sycl_impl.hpp.

#include "coll/algorithms/gather/sycl/gather_large_sycl_impl.hpp"
#include "coll/algorithms/gather/sycl/gather_large_sycl.hpp"

ccl::event gather_large(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        int root,
                        ccl_comm* comm,
                        std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        const ccl::vector_class<ccl::event>& deps) {
    // MT requires staging (IPC exchange is not thread safe). Local flag, not an
    // assignment into the shared env, which would race.
    const bool force_tmp_buf = comm->is_multi_thread_instance();
    if (force_tmp_buf) {
        LOG_DEBUG("invoking MT gather_large");
    }
    else {
        LOG_DEBUG("invoking gather_large");
    }

    sycl_ptrs_type sycl_ptrs;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const int comm_size = p2p_comm->size();

    bool is_tmp_used = force_tmp_buf || ccl::global_data::env().sycl_gather_tmp_buf ||
                       ccl::global_data::env().sycl_gather_large_tmp_buf;

    const bool read_protocol =
        ccl::global_data::env().sycl_gather_large_protocol == ccl_sycl_gather_protocol::read;

    if (!is_tmp_used) {
        // write exchanges root's recv_buf, read exchanges each rank's send_buf.
        // No registered-pointer fast path on write: recv_buf is root-only, so a
        // local get_registered_ptrs lookup would answer differently on root than
        // on the peers and split them across the collective exchange. Non-root
        // contributes send_buf: the API ignores its recv_buf, so it may be host
        // or freed memory that the exchange cannot take a handle on.
        const bool exchange_recv_buf = !read_protocol && p2p_comm->rank() == root;
        void* exchanged = exchange_recv_buf ? recv_buf : (void*)send_buf;
        std::vector<void*> ptrs{ exchanged }; // index 0
        auto [sched, exchange_entry] = do_ipc_exchange(p2p_comm.get(), global_stream, ptrs);

        if (read_protocol) {
            sycl_ptrs.node_ptrs_rd =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 0, exchanged, sched);
        }
        else {
            sycl_ptrs.node_ptrs_wr =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 0, exchanged, sched);
        }

        delete exchange_entry;
        delete sched;

        // Minimum-alignment gate for the IPC path, not the memcpy's vector width.
        // Below this floor we drop to staging. all_aligned also checks
        // (count * dsize) % align, which covers the rank offset.
        const size_t align = 4;
        void** gate_ptrs =
            read_protocol ? sycl_ptrs.node_ptrs_rd.data() : sycl_ptrs.node_ptrs_wr.data();
        if (!all_aligned(gate_ptrs, comm_size, count, dsize, align)) {
            LOG_DEBUG("gather_large: IPC pointers not 4-byte aligned, falling back to tmp-buf");
            is_tmp_used = true;
        }
    }

    return gather_large_impl(send_buf,
                             recv_buf,
                             count,
                             dtype,
                             root,
                             comm,
                             p2p_comm,
                             global_stream,
                             sycl_ptrs,
                             is_tmp_used,
                             deps);
}
