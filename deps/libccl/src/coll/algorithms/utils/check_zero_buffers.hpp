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

#include "comm/comm.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"

template <typename T>
void check_zero_buffers_cpu(LLPatternData ll_pattern_data,
                            const std::shared_ptr<ccl_comm> comm,
                            sycl::queue q,
                            T* ipcbuf0,
                            T* ipcbuf1,
                            size_t tmp_buf_size) {
    if (ll_pattern_data.should_zero_now()) {
        q.fill(ipcbuf0, 0, tmp_buf_size);
        q.fill(ipcbuf1, 0, tmp_buf_size);
        invoke_barrier(comm, q, {}, true);
    }
}

// Besides zeroing, this is the per-op cross-rank sync point for collectives
// on the recording path (MLSL-4992); do not skip it when nothing needs zeroing.
template <typename T>
void check_zero_buffers_gpu(LLPatternData ll_pattern_data,
                            const sycl::nd_item<1> it,
                            ccl_comm_barrier_data barrier_data,
                            T* ipcbuf0,
                            T* ipcbuf1,
                            size_t tmp_buf_size,
                            pattern_type type) {
    it.barrier();
    const bool zero_now = ll_pattern_data.should_zero_now();
    if (zero_now) {
        size_t per_item_count = tmp_buf_size / it.get_global_range()[0];
        size_t offset = per_item_count * it.get_global_linear_id();
        for (size_t i = offset; i < offset + per_item_count; ++i) {
            ipcbuf0[i] = 0;
            ipcbuf1[i] = 0;
        }
    }
    // The recording path (graph capture or CCL_SYCL_FORCE_RECORDING_PATH) reuses
    // tmp buffer 0 for every LL op (no rotation), so a collective waits until
    // every rank has finished the previous op before it writes into the peers'
    // slots (MLSL-4992). barrier_data spans the whole node comm, which a
    // send/recv cannot make every rank join, so send/recv sync here only after
    // zeroing. A send/recv right after a collective is not covered (MLSL-5007).
    if (zero_now || type == pattern_type::collective) {
        comm_barrier(barrier_data, it, true, true);
    }
}
