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

#include "oneapi/ccl.hpp"

class ccl_comm;
class ccl_stream;

// Single-domain small SYCL gather, push mode: every rank writes its block into
// root's tmp buffer at slot rank * count, comm barrier, then root copies the
// staged region into recv_buf in one transfer. The staged layout already matches
// recv_buf, so in-place needs no special case.
//
// Only valid while comm_size * count * dsize fits one tmp-buf generation;
// gather_scaleup_sycl checks that, and the impl CCL_THROW's if it is violated.
ccl::event gather_small(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        int root,
                        ccl_comm* comm,
                        std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        const ccl::vector_class<ccl::event>& deps);
