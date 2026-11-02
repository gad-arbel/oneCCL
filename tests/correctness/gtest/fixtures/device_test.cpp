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

#include "device_test.h"

#include "framework/test_runtime_environment.h"

#include <mpi.h>

namespace oneccl_test {

void DeviceTest::bring_up() {
    // Default: adopt the process's shared communicator, created once in
    // TestRuntimeEnvironment::SetUp. A suite that needs its own set — a
    // non-default num_comms, a non-null comm_config, or one that finalizes in
    // its body — creates and owns it here instead, so a body finalize never
    // reaches the process-wide communicator.
    if (num_comms() == 1 && comm_config() == nullptr &&
        finalize_in_teardown()) {
        comms_ = TestRuntimeEnvironment::comms();
        owns_comms_ = false;
    } else {
        // Set before the loop, so TearDown destroys a partial set.
        owns_comms_ = true;
        comms_.clear();
        for (int index = 0; index < num_comms(); ++index) {
            // An id names one communicator, and the process's id already
            // named the shared one.
            const onecclUniqueId id = env().mint_unique_id();
            onecclComm_t comm = nullptr;
            ASSERT_EQ(onecclSuccess,
                      env().init_world_comm(&comm, id, comm_config()))
                << "onecclCommInitRank failed";
            comms_.push_back(comm);
        }
    }

    // Assert the world every communicator was handed. Both halves earn their
    // place against the null plugin, which reports size 1 / rank 0 on every
    // rank — which is why occl_on_device cannot run null.
    for (onecclComm_t comm : comms_) {
        int size = 0;
        ASSERT_EQ(onecclSuccess, onecclCommCount(comm, &size));
        ASSERT_EQ(env().world_size(), size);
        int rank = -1;
        ASSERT_EQ(onecclSuccess, onecclCommUserRank(comm, &rank));
        ASSERT_EQ(env().world_rank(), rank);
    }
}

void DeviceTest::TearDown() {
    // Dispose of only the communicators this fixture itself created. The
    // shared set is finalized and destroyed once in
    // TestRuntimeEnvironment::TearDown.
    if (!owns_comms_) {
        return;
    }
    for (onecclComm_t comm : comms_) {
        if (finalize_in_teardown()) {
            EXPECT_EQ(onecclSuccess, onecclCommFinalize(comm));
        }
        EXPECT_EQ(onecclSuccess, onecclCommDestroy(comm));
    }
    comms_.clear();
}

void DeviceTest::run_threads(int n, const ThreadBody &body) {
    ASSERT_EQ(1, n) << "run_threads() supports only N == 1";
    ThreadContext ctx(env(), comms_);
    ASSERT_TRUE(body(ctx)) << "thread 0 failed";
}

bool DeviceTest::unanimous_agreement(bool local) {
    int mine = local ? 1 : 0;
    int world = 0;
    MPI_Allreduce(&mine, &world, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return world != 0;
}

bool DeviceTest::skip_if_unsupported(bool supported, const std::string &why) {
    if (!unanimous_agreement(supported)) {
        const ::testing::TestInfo *info =
            ::testing::UnitTest::GetInstance()->current_test_info();
        env().record_runtime_refusal(
            std::string(info->test_suite_name()) + "." + info->name(), why);
        // GTEST_SKIP expands to a void return, so it needs a void scope.
        [&why] { GTEST_SKIP() << why; }();
        return true;
    }
    return false;
}

} // namespace oneccl_test
