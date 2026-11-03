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

#include "coll/algorithms/alltoall/sycl/alltoall_large_sycl_impl.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/selection/policy/policy_env.hpp"

ccl::event alltoall_large(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          std::vector<size_t>& offsets,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          const std::shared_ptr<ccl_comm> p2p_comm,
                          ccl_stream* global_stream,
                          const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("invoking alltoall_large");

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();

    bool is_arc = is_arc_card(ccl::global_data::get().ze_data->devices[0].family);
    static const simple_chunk_size_table_t simple_chunk_sizes =
        parse_simple_chunk_size_spec(CCL_SYCL_ALLTOALL_SIMPLE_CHUNK_SIZE,
                                     ccl::global_data::env().sycl_alltoall_simple_chunk_size);
    // ARC does chunking as needed unless the simple-path spec overrides it
    // the policy knob `simple_chunk_size`, when set for this call, replaces the spec
    const size_t chunk_size =
        ccl::selection::call_env().sycl_alltoall_simple_chunk_size_override
            ? ccl::selection::call_env().sycl_alltoall_simple_chunk_size_override
            : !simple_chunk_sizes.empty()
                  ? select_simple_chunk_size(simple_chunk_sizes, count * dsize)
                  : (is_arc ? ccl::global_data::env().sycl_alltoall_chunking_threshold : 0);
    CCL_THROW_IF_NOT(!ccl::selection::call_env().sycl_copy_engine || is_arc,
                     "alltoall using copy engines not supported");

    // no constraints on rank reordering - performance should not be affected
    // end-to-end transfers are performed
    if (comm->is_multi_thread_instance() == true) {
        // LOG_DEBUG("|MT|: invoking alltoall_large");
        CCL_THROW("simple protocol with SPMD is not supported");
    }
    else {
        LOG_DEBUG("invoking alltoall_large");
    }

    auto lambda = [&]<typename T>() {
        return alltoall_large_impl<T>(
            send_buf, recv_buf, count, offsets, dtype, comm, p2p_comm, global_stream, deps, chunk_size);
    };
    return invoke_collective(lambda, dtype);
}
