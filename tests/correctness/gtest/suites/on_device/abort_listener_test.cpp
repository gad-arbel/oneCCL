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

// AbortListener::OnTestPartResult has two failure-exit paths
// (framework/abort_listener.cpp): MPI_Abort when MPI is usable, and _Exit when
// it is not (a fatal failure before MPI_Init or after MPI_Finalize). This test
// drives the _Exit path. A regression that skipped it would hang the whole
// ctest entry to its timeout instead of failing in seconds. It reuses the
// forked EXPECT_EXIT pattern of ConfigScopeTest (config_scope_test.cpp).
//
// The listener under test gets a probe that reports MPI as unusable, so the
// forked child never calls MPI. fork() copies a process whose libccl, MPI and
// Level-Zero threads hold locks, and a child that only calls _Exit takes none
// of them.

#include "executors/executor_factory.h"
#include "fixtures/device_test.h"
#include "framework/abort_listener.h"
#include "framework/thread_context.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>

using namespace oneccl_test;

namespace {

class AbortListenerDeathTest : public DeviceTest {};

TEST_F(AbortListenerDeathTest, exit_with_unusable_mpi_then_exits_directly) {
    EXPECT_EXIT(
        {
            AbortListener listener(/*mpi_usable=*/[] { return false; });
            const ::testing::TestPartResult fatal_result(
                ::testing::TestPartResult::kFatalFailure, __FILE__, __LINE__,
                "deliberate fatal failure with MPI unusable");
            listener.OnTestPartResult(fatal_result);

            // OnTestPartResult must not return here.
            std::exit(AbortListener::kAbortExitCode + 1);
        },
        ::testing::ExitedWithCode(AbortListener::kAbortExitCode), "")
        << "AbortListener::OnTestPartResult did not _Exit with "
           "kAbortExitCode when MPI was unusable";
}

class AbortListenerTest : public DeviceTest {};

// The MPI_Abort path. Rank 0 fails before the collective while its peer enters
// it. Run by hand at -n 2 with --gtest_also_run_disabled_tests: the job dies in
// seconds with the failure text visible, not at ctest's timeout.
TEST_F(AbortListenerTest,
       DISABLED_failing_rank_with_multiple_ranks_then_aborts_the_job) {
    if (skip_if_unsupported(env().world_size() > 1,
                            "needs a peer rank that enters the collective")) {
        return;
    }
    run_threads([](ThreadContext &ctx) {
        const CollectiveParams params(Collective::ONECCL_ALLREDUCE,
                                      onecclFloat32, PatternOp::ONECCL_SUM,
                                      /*count=*/1024);
        const std::unique_ptr<CollectiveExecutor> exec =
            ExecutorFactory::create(params, ctx);
        exec->setup();
        if (ctx.rank() == 0) {
            ADD_FAILURE() << "deliberate failure on rank 0 before the "
                             "collective";
            return false;
        }
        EXPECT_EQ(onecclSuccess, exec->execute(ctx.comm()));
        exec->verify();
        return true;
    });
}

} // namespace
