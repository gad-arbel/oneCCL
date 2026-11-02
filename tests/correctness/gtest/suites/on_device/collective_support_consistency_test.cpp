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

// ExecutorFactory's collective->executor switch (executor_factory.cpp) and
// ValidityTable's collective refusal rule (validity_table.cpp) are two
// independent "supported collectives" lists with no shared source of truth:
// adding a second collective means editing both, and missing one produces
// silent divergence - ValidityTable accepting a collective ExecutorFactory
// still returns nullptr for, or the reverse.
//
// This walks every Collective enumerator and pins that the two agree. It also
// covers the gap that no test asserted nullptr for any of the 8
// not-yet-implemented collectives at all.

#include "executors/executor_factory.h"
#include "fixtures/device_test.h"
#include "framework/thread_context.h"
#include "params/collective_params.h"
#include "params/device_capabilities.h"
#include "params/validity_table.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace oneccl_test;

namespace {

class CollectiveSupportConsistencyTest : public DeviceTest {};

TEST_F(CollectiveSupportConsistencyTest,
       validity_table_and_executor_factory_with_every_collective_then_agree) {
    run_threads([this](ThreadContext &ctx) {
        for (Collective collective : kAllCollectives) {
            // fp32/SUM/out-of-place isolates the "unsupported collective"
            // rule: it trips none of ValidityTable's other refusals (fp64,
            // bf16, AllToAll in-place), so is_valid()'s answer is purely about
            // the collective axis.
            CollectiveParams params(collective, onecclFloat32,
                                    PatternOp::ONECCL_SUM,
                                    /*count=*/4);

            std::string why;
            const bool validity_table_accepts = ValidityTable::is_valid(
                params, env().capabilities(), env().topology(), &why);

            const std::unique_ptr<CollectiveExecutor> executor =
                ExecutorFactory::create(params, ctx);
            const bool executor_factory_builds = (executor != nullptr);

            EXPECT_EQ(validity_table_accepts, executor_factory_builds)
                << "collective=" << static_cast<int>(collective)
                << ": ValidityTable::is_valid()=" << validity_table_accepts
                << " (" << why << ") but ExecutorFactory::create() "
                << (executor_factory_builds ? "built an executor"
                                            : "returned nullptr")
                << " - the two 'supported collectives' lists have diverged";
        }
        return true;
    });
}

} // namespace
