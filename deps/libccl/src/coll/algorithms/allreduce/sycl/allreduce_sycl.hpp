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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"

namespace ccl {

namespace v1 {
event allreduce_sycl(sycl::queue q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                     ccl_comm* comm,
                     ccl_stream* op_stream,
                     const allreduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done);

} // namespace v1

} // namespace ccl

ccl::event allreduce_small(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm* comm,
                           std::shared_ptr<ccl_comm> p2p_comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps);

ccl::event allreduce_large(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm* comm,
                           const std::shared_ptr<ccl_comm> p2p_comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps);

// LL protocols
template <template <typename, template <typename, int> class, bool, int> class Transmit>
ccl::event allreduce_ll(const void* src,
                        void* dst,
                        size_t count,
                        ccl::datatype dtype,
                        ccl::reduction reduction,
                        ccl_comm* comm,
                        const std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        bool& done);

ccl::event allreduce_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* comm,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   sycl_scaleout_tune_attr tune_attr,
                                   bool& done,
                                   bool is_cpu_buffers = false,
                                   bool is_remainder_chunk = false);
