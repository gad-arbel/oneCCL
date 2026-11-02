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

#include "collective_test.h"

#include "executors/executor_factory.h"

#include <memory>
#include <mpi.h>

namespace oneccl_test {

void CollectiveTest::run_one() {
    run_threads([this](ThreadContext &ctx) {
        return run_step(ctx, onecclSuccess, /*verify=*/true);
    });
}

void CollectiveTest::run_one_expecting(onecclResult_t expected) {
    // No verify(): a refused operation left the recv buffer poisoned.
    run_threads([this, expected](ThreadContext &ctx) {
        return run_step(ctx, expected, /*verify=*/false);
    });
}

bool CollectiveTest::run_step(ThreadContext &ctx, onecclResult_t expected,
                              bool verify) {
    const std::unique_ptr<CollectiveExecutor> exec =
        ExecutorFactory::create(params(), ctx);
    EXPECT_NE(exec, nullptr) << "no executor for this collective";
    if (exec == nullptr) {
        return false;
    }

    // iterations defaults to 1, so the loop degenerates to the ordinary path.
    // Out of place, every iteration writes the same expected values, so
    // verify() runs once at the end. In place, each iteration reduces the
    // previous result, so setup() refreshes the input before each one.
    for (size_t iteration = 0; iteration < params().iterations; ++iteration) {
        if (iteration == 0 || params().is_in_place) {
            if (iteration > 0) {
                // The refill overwrites the buffer a peer may still be reading
                // through IPC, so every rank's queue drains first. One thread
                // per rank, as run_threads() enforces, and one group spanning
                // the world, the default thread_groups().
                if (ctx.has_device()) {
                    ctx.queue().wait();
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            // setup() asserts in place, which returns only from setup().
            exec->setup();
            if (HasFatalFailure()) {
                return false;
            }
        }
        const onecclResult_t code = exec->execute(ctx.comm());
        EXPECT_EQ(expected, code);
        if (code != expected) {
            return false;
        }
    }
    if (verify) {
        exec->verify();
    }
    return true;
}

} // namespace oneccl_test
