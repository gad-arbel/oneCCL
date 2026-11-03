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

#include "buffers/device_buffer.h"
#include "data/source_map.h"
#include "data/test_data.h"
#include "framework/thread_context.h"
#include "params/collective_params.h"

#include "oneapi/ccl.h"
#include <memory>

namespace oneccl_test {

// One class per collective, and the only place in the tree that knows which
// collective is being run. Owns the API call and the buffers it reads and
// writes; owns no verification logic and no arithmetic.
class CollectiveExecutor {
  public:
    CollectiveExecutor(const CollectiveParams &params, ThreadContext &ctx);
    virtual ~CollectiveExecutor() = default;

    // ITestData::create, fill_send, poison_recv, H2D copy, buffer sizing.
    virtual void setup() = 0;

    // The collective entry point, its code returned unexamined — the caller
    // chooses which code to expect. ASSERT_* compiles only in a void function,
    // so execute() cannot assert.
    virtual onecclResult_t execute(onecclComm_t comm) = 0;

    // Shared window loop over recv_buff_: one windowed D2H copy per window,
    // then data_->verify().
    virtual void verify();

    // Which ranks contributed to each receive element, and at which index in
    // that rank's send region.
    virtual SourceMap source_map() const = 0;

  protected:
    // A device buffer on the context's queue, or a host-backed one where the
    // run has no device. The only place an executor makes that choice.
    std::unique_ptr<DeviceBuffer> make_buffer(size_t bytes) const;

    // Elements per window for the windowed D2H copy in verify().
    static constexpr size_t kWindowElementCount =
        ITestData::kWindowElementCount;

    CollectiveParams params_;
    ThreadContext &ctx_;
    std::unique_ptr<ITestData> data_;
    std::unique_ptr<DeviceBuffer> send_buff_;
    std::unique_ptr<DeviceBuffer> recv_buff_;
};

// Bytes per element for a oneCCL data type.
size_t dtype_size(onecclDataType_t dtype);

} // namespace oneccl_test
