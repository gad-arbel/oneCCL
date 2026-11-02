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

#include "collective_executor.h"

#include "gtest/gtest.h"

#include <algorithm>

namespace oneccl_test {

size_t dtype_size(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclInt8:
    case onecclUint8:
        return 1;
    case onecclFloat16:
    case onecclBfloat16:
        return 2;
    case onecclInt32:
    case onecclUint32:
    case onecclFloat32:
        return 4;
    case onecclInt64:
    case onecclUint64:
    case onecclFloat64:
        return 8;
    default:
        return 0;
    }
}

CollectiveExecutor::CollectiveExecutor(const CollectiveParams &params,
                                       ThreadContext &ctx)
    : params_(params), ctx_(ctx) {}

std::unique_ptr<DeviceBuffer>
CollectiveExecutor::make_buffer(size_t bytes) const {
    if (ctx_.has_device()) {
        return std::make_unique<DeviceBuffer>(bytes, ctx_.queue());
    }
    return std::make_unique<DeviceBuffer>(bytes);
}

void CollectiveExecutor::verify() {
    // Non-fatal: verify() and every comparison below it record and return,
    // so a mismatch is aggregated in OnTestEnd rather than aborting the job.
    const size_t size_of_element = dtype_size(params_.dtype);
    if (size_of_element == 0) {
        ADD_FAILURE() << "unknown dtype in verify(): "
                      << static_cast<int>(params_.dtype);
        return;
    }
    const size_t total_element_cnt = recv_buff_->bytes() / size_of_element;

    // Window loop over the receive buffer: copy each window to host and
    // verify it.
    for (size_t window_offset = 0; window_offset < total_element_cnt;
         window_offset += kWindowElementCount) {
        const size_t window_element_cnt =
            std::min(kWindowElementCount, total_element_cnt - window_offset);
        const HostBuffer window =
            recv_buff_->d2h_copy(window_offset * size_of_element,
                                 window_element_cnt * size_of_element);
        // window_offset is absolute, so source_map() is asked nothing about
        // windows.
        data_->verify(window, source_map(), window_offset);
    }
}

} // namespace oneccl_test
