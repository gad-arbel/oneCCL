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

// Resolves ccl::datatype to a concrete element type and forwards to
// gather_small_impl<T>; the algorithm lives in gather_small_sycl_impl.hpp.

#include "coll/algorithms/gather/sycl/gather_small_sycl_impl.hpp"
#include "coll/algorithms/gather/sycl/gather_small_sycl.hpp"

ccl::event gather_small(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        int root,
                        ccl_comm* comm,
                        std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG(comm->is_multi_thread_instance() ? "|MT| " : "", "invoking gather_small");

    auto lambda = [&]<typename T>() {
        return gather_small_impl<T>(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps);
    };

    return invoke_collective(lambda, dtype);
}
