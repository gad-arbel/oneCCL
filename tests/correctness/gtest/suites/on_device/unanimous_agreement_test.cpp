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

// DeviceTest::unanimous_agreement() MIN-reduces a flag across the world, which
// is what lets skip_if_unsupported() skip every rank or none. The test calls
// unanimous_agreement() directly, so it runs and exits like any other entry. It
// also checks the refusal record that skip_if_unsupported() writes.

#include "fixtures/device_test.h"

#include <gtest/gtest.h>

using namespace oneccl_test;

namespace {

class UnanimousAgreementTest : public DeviceTest {};

// Rank 1 passes false, so every rank, rank 0 included, must see false.
TEST_F(UnanimousAgreementTest,
       unanimous_agreement_with_one_rank_false_then_false) {
    ASSERT_GE(env().world_size(), 2) << "run with at least two ranks";
    EXPECT_FALSE(unanimous_agreement(env().world_rank() != 1));
}

TEST_F(UnanimousAgreementTest,
       unanimous_agreement_with_every_rank_true_then_true) {
    EXPECT_TRUE(unanimous_agreement(true));
}

// skip_if_unsupported() records through this before it skips. The skip itself
// cannot be asserted from the skipped test.
TEST_F(UnanimousAgreementTest,
       runtime_refusal_with_reason_then_recorded_with_that_reason) {
    const size_t before = env().runtime_refusals().size();
    env().record_runtime_refusal("Suite.Test", "no onecclMemAlloc");
    ASSERT_EQ(before + 1, env().runtime_refusals().size());
    EXPECT_EQ("Suite.Test", env().runtime_refusals().back().test);
    EXPECT_EQ("no onecclMemAlloc", env().runtime_refusals().back().why);
    env().clear_runtime_refusals();
}

} // namespace
