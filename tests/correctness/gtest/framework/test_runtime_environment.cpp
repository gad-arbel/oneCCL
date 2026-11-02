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

#include "test_runtime_environment.h"

#include "test_env.h"

#include <mpi.h>

namespace oneccl_test {

std::vector<onecclComm_t> TestRuntimeEnvironment::comms_;
std::thread::id TestRuntimeEnvironment::main_thread_;

void TestRuntimeEnvironment::SetUp() {
    main_thread_ = std::this_thread::get_id();

    // The process's shared communicator, created once here rather than once
    // per test. One communicator over the whole world; the multi-communicator
    // group cases raise the count.
    const TestEnv &env = TestEnv::instance();
    onecclComm_t comm = nullptr;
    ASSERT_EQ(onecclSuccess,
              env.init_world_comm(&comm, env.unique_id(), /*config=*/nullptr))
        << "onecclCommInitRank failed";
    comms_.push_back(comm);
}

void TestRuntimeEnvironment::TearDown() {
    // Finalize then destroy every shared communicator before MPI_Finalize, so
    // MPI outlives the destroy.
    for (onecclComm_t comm : comms_) {
        EXPECT_EQ(onecclSuccess, onecclCommFinalize(comm));
        EXPECT_EQ(onecclSuccess, onecclCommDestroy(comm));
    }
    comms_.clear();
    MPI_Finalize();
}

const std::vector<onecclComm_t> &TestRuntimeEnvironment::comms() {
    return comms_;
}

bool TestRuntimeEnvironment::on_main_thread() {
    return std::this_thread::get_id() == main_thread_;
}

} // namespace oneccl_test
