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

#include "device_test.h"
#include "framework/thread_context.h"
#include "params/collective_params.h"

#include <gtest/gtest.h>

namespace oneccl_test {

// One operation per test. The two run entry points turn the executor's result
// code into a verdict; which verdict differs by suite.
class CollectiveTest : public DeviceTest,
                       public ::testing::WithParamInterface<CollectiveParams> {
  protected:
    const CollectiveParams &params() const { return GetParam(); }

    // setup(), then iterations × execute() each expected onecclSuccess, then
    // one verify().
    void run_one();

    // Expect a named non-success code and skip verify() — a refused operation
    // left the recv buffer poisoned, so there is no result to compare.
    void run_one_expecting(onecclResult_t expected);

  private:
    // A local executor, setup(), iterations × execute(), then verify() if
    // asked. Checks with EXPECT_* and returns false, never ASSERT_*: the step
    // is the same code at one thread and at N, and ASSERT_* off the main
    // thread ends only that thread. False means the step stopped before the
    // end of its sequence. A verify() mismatch is recorded as a non-fatal
    // failure and still returns true, so OnTestEnd aggregates it rather than
    // AbortListener aborting the job.
    bool run_step(ThreadContext &ctx, onecclResult_t expected, bool verify);
};

} // namespace oneccl_test
