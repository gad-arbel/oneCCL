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

#include "coll/algorithms/allgatherv/sycl/allgatherv_small_sycl_impl.hpp"

ccl::event allgatherv_small(sycl::queue& q,
                            const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            const ccl::vector_class<size_t>& offsets,
                            ccl::datatype dtype,
                            ccl_comm* comm,
                            ccl_stream* global_stream,
                            const ccl::vector_class<ccl::event>& deps,
                            bool& done) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking allgatherv_small");
    }
    else {
        LOG_DEBUG("invoking allgatherv_small");
    }

    auto lambda = [&]<typename T>() {
        return allgatherv_small_impl<T>(
            q, send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps, done);
    };
    return invoke_collective(lambda, dtype);
}
