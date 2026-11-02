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

#include <memory>
#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp" // for sycl_ptrs_type

// Forward declarations
class ccl_comm;
class ccl_stream;

/*
 * Level Zero Broadcast using Copy Engine, for large payloads, including the IPC exchange
 * of the peers' recv buffers it needs. Call only when the broadcast scale-up resolver
 * selected the l0_copy_engine algo.
 */
ccl::event broadcast_large_ze_copy_engine_with_exchange(sycl::queue& q,
                                                        const void* send_buf,
                                                        void* recv_buf,
                                                        size_t count,
                                                        ccl::datatype dtype,
                                                        int root,
                                                        ccl_comm* comm,
                                                        std::shared_ptr<ccl_comm> p2p_comm,
                                                        ccl_stream* global_stream,
                                                        const ccl::vector_class<ccl::event>& deps);
