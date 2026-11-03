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

// The on-device correctness baseline for AllReduce. One shared TEST_P body on
// CollectiveTest, plus one INSTANTIATE_TEST_SUITE_P per (count list) — the
// prefix names the collective, which is what lets a per-collective ctest
// filter select its rows.

#include "data/test_data.h"
#include "fixtures/collective_test.h"
#include "framework/test_env.h"
#include "params/param_expander.h"

#include <gtest/gtest.h>
#include <vector>

using namespace oneccl_test;

namespace {

const std::vector<onecclDataType_t> kDtypes = {onecclFloat32, onecclBfloat16,
                                               onecclInt32};
const std::vector<PatternOp> kOps = {PatternOp::ONECCL_SUM,
                                     PatternOp::ONECCL_MAX};

// kWindowElementCount + 33 exercises a remainder tail that crosses a window
// boundary (CollectiveExecutor::verify()'s windowed D2H loop and
// TestDataImpl::compare()'s window loop) - the other three counts stay well
// below one window.
const std::vector<size_t> kRemainderCounts = {
    0, 1, 9, 33, ITestData::kWindowElementCount + 33};
const std::vector<size_t> kAlignedCounts = {1024, 1024 * 1024}; // 1 Ki, 1 Mi

std::vector<CollectiveParams>
expand_allreduce(const std::vector<size_t> &counts) {
    // Grab the process-global test environment (device capabilities +
    // topology).
    const TestEnv &env = TestEnv::instance();
    // Cross-product AllReduce x kDtypes x kOps x counts, filtered by
    // capabilities/topology.
    return ParamExpander::expand(Collective::ONECCL_ALLREDUCE, kDtypes, kOps,
                                 counts, env.capabilities(), env.topology(),
                                 &TestEnv::instance().expansion_refusals());
}

// The same rows in place: one buffer, send == recv.
std::vector<CollectiveParams> expand_allreduce_in_place() {
    std::vector<size_t> counts = kRemainderCounts;
    counts.insert(counts.end(), kAlignedCounts.begin(), kAlignedCounts.end());
    const TestEnv &env = TestEnv::instance();
    return ParamExpander::expand(Collective::ONECCL_ALLREDUCE, kDtypes, kOps,
                                 counts, env.capabilities(), env.topology(),
                                 &TestEnv::instance().expansion_refusals(),
                                 /*is_in_place=*/true);
}

std::string param_name(const ::testing::TestParamInfo<CollectiveParams> &info) {
    const CollectiveParams &params = info.param;
    const std::string dtype = dtype_name(params.dtype);
    return dtype + "_" + op_name(params.op) + "_count" +
           std::to_string(params.count);
}

} // namespace

// Defines the test pattern: a class derived from the CollectiveTest fixture,
// registered under the suite name CollectiveTest. On its own it produces zero
// runnable tests — it is a template waiting for parameters. The body reads the
// current param via params() and runs+validates one AllReduce for it.
TEST_P(CollectiveTest,
       allreduce_with_each_param_row_then_result_matches_expected) {
    run_one();
}

// Generates the runnable tests: for every value in the parameter set, stamp out
// an instance of every TEST_P registered under the same suite name
// (CollectiveTest). The prefix (AlignedCounts_AllReduce) becomes part of each
// test name so a per-collective ctest filter can select these rows;
// expand_allreduce cross-products the count list with kDtypes x kOps (filtered
// by device capabilities/topology), ValuesIn makes one instance per element,
// and param_name renders each param into a readable suffix (e.g.
// fp32_sum_count1024).
INSTANTIATE_TEST_SUITE_P(AlignedCounts_AllReduce, CollectiveTest,
                         ::testing::ValuesIn(expand_allreduce(kAlignedCounts)),
                         param_name);

// Same body, second instantiation over the remainder-path counts.
INSTANTIATE_TEST_SUITE_P(
    RemainderCounts_AllReduce, CollectiveTest,
    ::testing::ValuesIn(expand_allreduce(kRemainderCounts)), param_name);

// Same body, both count lists in place.
INSTANTIATE_TEST_SUITE_P(InPlace_AllReduce, CollectiveTest,
                         ::testing::ValuesIn(expand_allreduce_in_place()),
                         param_name);
