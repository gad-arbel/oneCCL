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

// Single-domain large SYCL gather. Three data paths, picked at runtime:
//
//  - IPC write (default): each rank pushes its block into root's recv_buf.
//  - IPC read (CCL_SYCL_GATHER_LARGE_PROTOCOL=read): root pulls each rank's
//    block out of that rank's send_buf. Same bytes, different placement of the
//    traffic; which wins depends on the fabric.
//  - tmp-buf staged, chunked. Used when IPC pointers are unavailable or
//    misaligned, for multi-thread instances, or when CCL_SYCL_GATHER_TMP_BUF /
//    CCL_SYCL_GATHER_LARGE_TMP_BUF force it.
ccl::event gather_large(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        int root,
                        ccl_comm* comm,
                        std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        const ccl::vector_class<ccl::event>& deps);
