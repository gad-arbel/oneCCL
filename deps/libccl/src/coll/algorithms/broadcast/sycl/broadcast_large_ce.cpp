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

#include "coll/algorithms/broadcast/sycl/broadcast_large_ce.hpp"

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/global/global.hpp"
#include "common/ze/ze_wrapper.hpp"
#include "common/utils/ce_utils.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp" // do_ipc_exchange, get_ipc_ptrs

// Level Zero Broadcast using Copy Engine, for large payloads
//
// Root push protocol: the root writes its send_buf into every peer's recv_buf over the
// copy engine and signals that peer, while each non-root rank only waits for the root's
// signal. Unlike allgather/alltoall, only one rank issues copies here - a broadcast has a
// single source, so there is no per-rank offset and every destination gets the full payload.
//
// Note the root does not wait for peers to consume the data, matching the allgather CE path:
// the event returned to the root completes once its own copies are done.
static ccl::event broadcast_large_ze_copy_engine(sycl::queue& q,
                                                 const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 int root,
                                                 ccl_comm* comm,
                                                 ccl_stream* global_stream,
                                                 sycl_ptrs_type& sycl_ptrs,
                                                 const ccl::vector_class<ccl::event>& deps) {
    // 1. run prologue - sync and get cmd list
    LOG_TRACE("start - rank(", comm->rank(), ")");

    ce_utils_t ce_utils(comm, q);
    ze_cmd_list_ptr cmd_list = ce_utils.prolog(comm, q);
    CCL_THROW_IF_NOT(cmd_list, "failed to create command list for copy engine");

    // 2. copy data from root to all ranks over L0 list

    // get data type size
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    // comm and data size needed for steps calculations
    int my_node_rank = comm->get_node_comm()->rank();
    size_t comm_size = comm->get_node_comm()->size();
    size_t data_size = count * dsize;
    LOG_DEBUG("global rank(",
              comm->rank(),
              "), node rank(",
              my_node_rank,
              "/",
              comm_size,
              "), data size(",
              data_size,
              "), root(",
              root,
              "), send from {",
              send_buf,
              "}, recv to {",
              recv_buf,
              "}");

    if (my_node_rank == root) {
        // place the payload in our own recv buffer, nothing to do when in-place
        if (send_buf != recv_buf) {
            LOG_TRACE(
                "copy self data ", data_size, " bytes from {", send_buf, "} to {", recv_buf, "}");
            CCL_THROW_IF_NOT(cmd_list->mem_copy(recv_buf, send_buf, data_size) == ZE_RESULT_SUCCESS,
                             "failed to copy self data on rank ",
                             my_node_rank);
        }

        // comm size - 1 steps
        uint32_t steps = comm_size - 1;

        // issue every send before waiting on anything, so the copies overlap on the engine
        for (size_t step = 0; step < steps; step++) {
            // Step k: Send to (my + k + 1)
            int to_rank = (my_node_rank + step + 1) % comm_size;

            // copy destination is to_rank recv address, the full payload with no offset
            void* dest_addr = sycl_ptrs.node_ptrs_wr[to_rank];
            LOG_DEBUG("Rank(",
                      my_node_rank,
                      ") copy ",
                      data_size,
                      " bytes from {",
                      send_buf,
                      "} to {",
                      dest_addr,
                      ") on rank(",
                      to_rank,
                      ")");

            // get the signal event
            ze_cb_event_ptr remote_ev = comm->get_node_comm()->get_remote_cb_event(to_rank);

            // send to to_rank and signal remote event
            CCL_THROW_IF_NOT(
                cmd_list->mem_copy(dest_addr, send_buf, data_size) == ZE_RESULT_SUCCESS,
                "failed to copy to rank ",
                to_rank);
            LOG_TRACE("signal to rank(",
                      to_rank,
                      "), remote event ptr {",
                      remote_ev,
                      "}, handle {",
                      remote_ev->handle(),
                      "}");
            CCL_THROW_IF_NOT(remote_ev->signal(*cmd_list) == ZE_RESULT_SUCCESS,
                             "failed to signal remote cb event for rank ",
                             to_rank);
        }
    }
    else {
        // get wait event and wait for recv from the root
        ze_cb_event_ptr local_ev = comm->get_node_comm()->get_local_cb_event(root);
        LOG_TRACE("add wait to recv from root(",
                  root,
                  "), local event ptr {",
                  local_ev,
                  "}, handle {",
                  local_ev->handle(),
                  "}");
        CCL_THROW_IF_NOT(cmd_list->wait_event(*local_ev) == ZE_RESULT_SUCCESS,
                         "failed to wait on cb event from root ",
                         root);

        // Reset the counter after waiting to prepare for next collective
        CCL_THROW_IF_NOT(local_ev->reset_counter(*cmd_list) == ZE_RESULT_SUCCESS,
                         "failed to reset local cb event for root ",
                         root);
    }

    // 3. run epilogue - sync and signal completion
    std::shared_ptr<sycl::event> sycl_ev = ce_utils.epilog(comm, q);
    LOG_TRACE("all ranks synchronized - rank ", my_node_rank);
    ccl::event ccl_ev = ccl::event::create_from_native(*sycl_ev);
    return ccl_ev;
}

// Broadcast over the L0 copy engine, including the IPC exchange it needs.
//
// Only the remote recv buffers are exchanged: in the root push protocol the peers never read
// the root's send buffer, so exchanging it would be wasted work.
ccl::event broadcast_large_ze_copy_engine_with_exchange(sycl::queue& q,
                                                        const void* send_buf,
                                                        void* recv_buf,
                                                        size_t count,
                                                        ccl::datatype dtype,
                                                        int root,
                                                        ccl_comm* comm,
                                                        std::shared_ptr<ccl_comm> p2p_comm,
                                                        ccl_stream* global_stream,
                                                        const ccl::vector_class<ccl::event>& deps) {
    sycl_ptrs_type sycl_ptrs;

    std::vector<void*> ptrs{ recv_buf }; // index 0
    auto [sched, exchange_entry] = do_ipc_exchange(p2p_comm.get(), global_stream, ptrs);

    sycl_ptrs.node_ptrs_wr = get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 0, recv_buf, sched);

    delete exchange_entry;
    delete sched;

    return broadcast_large_ze_copy_engine(
        q, send_buf, recv_buf, count, dtype, root, comm, global_stream, sycl_ptrs, deps);
}
