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

#include "abort_listener.h"

#include "test_runtime_environment.h"

#include <cstdlib>
#include <mpi.h>

namespace oneccl_test {

const int AbortListener::kAbortExitCode = 1;

bool AbortListener::mpi_is_usable() {
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    return initialized != 0 && finalized == 0;
}

void AbortListener::OnTestPartResult(const ::testing::TestPartResult &result) {
    // Fatal only: ASSERT_* diverges control flow, so the failing rank has
    // already left the sequence of collectives its peers are still executing
    // and can never join an aggregating one. A non-fatal failure runs the body
    // to completion and is aggregated in OnTestEnd instead.
    if (!result.fatally_failed()) {
        return;
    }

    // Off the main thread the part is recorded and nothing is aborted; the
    // worker's bool carries the verdict up. MPI_Abort from a worker races a
    // peer's.
    if (!TestRuntimeEnvironment::on_main_thread()) {
        return;
    }

    if (mpi_usable_()) {
        MPI_Abort(MPI_COMM_WORLD, kAbortExitCode);
    }
    // A failure before MPI_Init or after MPI_Finalize still ends the process.
    // _Exit, not exit: static destructors in a process whose MPI is gone are
    // the code most likely to hang.
    std::_Exit(kAbortExitCode);
}

void AbortListener::OnTestEnd(const ::testing::TestInfo & /*info*/) {
    // Every test is a collective: one bit — did this rank record a non-fatal
    // failure during this test — reduced with MAX across the world. Every rank
    // whose bit was 0 calls ADD_FAILURE(), so one rank's data mismatch fails
    // the test on all of them.
    if (!mpi_usable_()) {
        return;
    }

    int local_failed =
        ::testing::UnitTest::GetInstance()->current_test_info() != nullptr &&
                ::testing::Test::HasNonfatalFailure()
            ? 1
            : 0;
    int world_failed = 0;
    MPI_Allreduce(&local_failed, &world_failed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (world_failed != 0 && local_failed == 0) {
        ADD_FAILURE() << "peer rank recorded a non-fatal failure for this test";
    }
}

} // namespace oneccl_test
