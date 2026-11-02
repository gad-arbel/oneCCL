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

#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_small_sycl_impl.hpp"

ccl::event reduce_scatter_small(const void* send_buf,
                                void* recv_buf,
                                size_t recv_count,
                                size_t rem_count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm* comm,
                                std::shared_ptr<ccl_comm> p2p_comm,
                                ccl_stream* global_stream,
                                const ccl::vector_class<ccl::event>& deps) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking reduce_scatter_small");
    }
    else {
        LOG_DEBUG("invoking reduce_scatter_small");
    }

    auto lambda = [&]<typename T>() {
        return reduce_scatter_small_impl<T>(
            send_buf, recv_buf, recv_count, rem_count, dtype, reduction, comm, p2p_comm, global_stream, deps);
    };
    return invoke_collective(lambda, dtype);
}
