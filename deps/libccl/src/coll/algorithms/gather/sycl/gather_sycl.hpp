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

// SYCL gather entry points. gather_sycl resolves the p2p domain and routes
// single-domain calls to gather_scaleup_sycl; multi-domain is unimplemented and
// always reports done=false, which hands the call to the `direct` algorithm.
//
// Contract: every rank contributes `count` elements; root returns
// comm_size * count with rank i at offset i * count. recv_buf is unused on
// non-root ranks and may be nullptr there.

namespace ccl {
namespace v1 {

event gather_scaleup_sycl(sycl::queue& q,
                          const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          datatype dtype,
                          int root,
                          ccl_comm* comm,
                          std::shared_ptr<ccl_comm> p2p_comm,
                          ccl_stream* global_stream,
                          const vector_class<event>& deps,
                          bool& done);

event gather_sycl_multi_domain(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* global_comm,
                               std::shared_ptr<ccl_comm> p2p_comm,
                               std::shared_ptr<ccl_comm> r2r_comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done);

event gather_sycl(sycl::queue q,
                  const void* send_buf,
                  void* recv_buf,
                  size_t count,
                  datatype dtype,
                  int root,
                  ccl_comm* global_comm,
                  ccl_stream* global_stream,
                  const gather_attr& attr,
                  const vector_class<event>& deps,
                  bool& done);

} // namespace v1
} // namespace ccl
