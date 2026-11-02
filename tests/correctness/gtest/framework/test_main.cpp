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

// The tree's own main: MPI init and the whole TestEnv before InitGoogleTest,
// so parameter generators expand against the real world size and measured
// capabilities. This main carries MPI_Init_thread; the device
// binaries link gtest rather than gtest_main.

#include "abort_listener.h"
#include "refusal_report.h"
#include "test_env.h"
#include "test_runtime_environment.h"

#include <cstdlib>
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {

    int provided = 0;
    // Request MPI_THREAD_MULTIPLE so that any thread may call MPI concurrently.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // The threaded family and AbortListener both need MPI_THREAD_MULTIPLE; a
    // lower level cannot support a listener entered from a worker thread.
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI granted thread level " << provided << ", need "
                  << MPI_THREAD_MULTIPLE << " (MPI_THREAD_MULTIPLE)\n";
        MPI_Finalize();
        return 1;
    }

    // World split, onecclSetDevice, device, queue, capabilities_, topology_,
    // unique_id_ — all reduced across the world. InitGoogleTest expands the
    // parameter generators, so TestEnv must be fully built before it.
    oneccl_test::TestEnv::create();
    const oneccl_test::TestEnv &env = oneccl_test::TestEnv::instance();

    const oneccl_test::LaunchTopology &topology = env.topology();
    // Print the measured topology on every run, so the log records the world
    // the run actually got.
    if (env.world_rank() == 0) {
        std::cout << "[topology] ranks_per_node=" << topology.ranks_per_node
                  << " nodes=" << topology.nodes
                  << " world_size=" << env.world_size() << '\n';
    }

    ::testing::InitGoogleTest(&argc, argv);

    // Fail-fast retires one rank's remaining schedule and not its peers', and
    // shuffle seeds each rank's order from its own clock. Both desynchronise
    // the OnTestEnd aggregation into a hang. Refuse them above one rank; a
    // single-rank run has no peer to desynchronise from and keeps them.
    if (env.world_size() > 1 &&
        (GTEST_FLAG_GET(fail_fast) || GTEST_FLAG_GET(shuffle))) {
        std::cerr << "--gtest_fail_fast and --gtest_shuffle are refused when "
                     "world_size > 1: they desynchronise the cross-rank "
                     "failure aggregation\n";
        MPI_Finalize();
        return 1;
    }

    // The environment's TearDown finalizes MPI, and MPI cannot start again, so
    // recreating the environment for a repeat uses MPI after finalize.
    if (GTEST_FLAG_GET(recreate_environments_when_repeating)) {
        std::cerr << "--gtest_recreate_environments_when_repeating is "
                     "refused: the environment's TearDown finalizes MPI\n";
        MPI_Finalize();
        return 1;
    }

    // Add the test runtime environment to the global test environment.
    ::testing::AddGlobalTestEnvironment(
        new oneccl_test::TestRuntimeEnvironment);
    // Add the abort listener to the test framework.
    ::testing::UnitTest::GetInstance()->listeners().Append(
        new oneccl_test::AbortListener);

    const int result = RUN_ALL_TESTS();

    // Every rank holds the same refusals, so rank 0 reports them.
    // params/ has no MPI, so the report is written here.
    if (env.world_rank() == 0) {
        oneccl_test::print_refusal_report(env);
    }

    // gtest skips the environment's TearDown when no test runs (a filter that
    // matches nothing, --gtest_list_tests), so MPI_Finalize did not run.
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized == 0) {
        MPI_Finalize();
    }
    return result;
}
