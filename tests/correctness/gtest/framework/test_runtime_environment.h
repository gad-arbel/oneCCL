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

#include "oneapi/ccl.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace oneccl_test {

// Creates the process's shared communicators in SetUp and nothing else; holds
// MPI_Finalize in TearDown, so MPI outlives every onecclCommDestroy. That
// teardown ordering is the latch the testing::Environment exists for.
class TestRuntimeEnvironment : public ::testing::Environment {
  public:
    void SetUp() override;    // called before any tests are run
    void TearDown() override; // called after all tests are run

    // The shared communicators, adopted by DeviceTest::bring_up.
    static const std::vector<onecclComm_t> &comms();

    // The main thread's identity, recorded in SetUp. AbortListener compares
    // against it so a worker's MPI_Abort never races a peer's.
    static bool on_main_thread();

  private:
    static std::vector<onecclComm_t> comms_;
    static std::thread::id main_thread_;
};

} // namespace oneccl_test
