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

// ValidityTable and ParamExpander are pure, device-free functions. These tests
// inject DeviceCapabilities combinations that allreduce_test.cpp's
// runtime-filtered instantiation cannot, and check the `why` string and
// ParamExpander::expand's refusals out-param.

#include "params/collective_params.h"
#include "params/device_capabilities.h"
#include "params/param_expander.h"
#include "params/validity_table.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace oneccl_test;

namespace {

// A capable device / single-node topology, with nothing in the params that
// any rule refuses - the baseline every single-rule test perturbs one field
// away from.
DeviceCapabilities all_capable() {
    return DeviceCapabilities{/*supports_fp64=*/true, /*supports_bf16=*/true,
                              /*devices_per_node=*/1};
}

} // namespace

// fp64 refused when the device does not support it.
TEST(ValidityTableTest,
     validate_with_fp64_and_unsupported_device_then_refuses) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclFloat64,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    DeviceCapabilities capabilities = all_capable();
    capabilities.supports_fp64 = false;
    LaunchTopology topology;

    std::string why;
    EXPECT_FALSE(ValidityTable::is_valid(params, capabilities, topology, &why));
    EXPECT_EQ(why, kWhyFp64Unsupported);
}

TEST(ValidityTableTest, validate_with_fp64_and_supported_device_then_allows) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclFloat64,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    LaunchTopology topology;

    std::string why;
    EXPECT_TRUE(ValidityTable::is_valid(params, all_capable(), topology, &why))
        << why;
}

// bf16 refused when the device does not support it.
TEST(ValidityTableTest,
     validate_with_bf16_and_unsupported_device_then_refuses) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclBfloat16,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    DeviceCapabilities capabilities = all_capable();
    capabilities.supports_bf16 = false;
    LaunchTopology topology;

    std::string why;
    EXPECT_FALSE(ValidityTable::is_valid(params, capabilities, topology, &why));
    EXPECT_EQ(why, kWhyBf16Unsupported);
}

TEST(ValidityTableTest, validate_with_bf16_and_supported_device_then_allows) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclBfloat16,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    LaunchTopology topology;

    std::string why;
    EXPECT_TRUE(ValidityTable::is_valid(params, all_capable(), topology, &why))
        << why;
}

// fp32 SUM AllReduce in place passes every rule on a capable device, which the
// in-place allreduce rows depend on.
TEST(ValidityTableTest, validate_with_allreduce_and_in_place_then_allows) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclFloat32,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    params.is_in_place = true;
    LaunchTopology topology;

    std::string why;
    EXPECT_TRUE(ValidityTable::is_valid(params, all_capable(), topology, &why))
        << why;
}

// AllToAll in-place is refused unconditionally.
TEST(ValidityTableTest, validate_with_alltoall_and_in_place_then_refuses) {
    CollectiveParams params(Collective::ONECCL_ALLTOALL, onecclFloat32,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    params.is_in_place = true;
    LaunchTopology topology;

    std::string why;
    EXPECT_FALSE(
        ValidityTable::is_valid(params, all_capable(), topology, &why));
    EXPECT_EQ(why, kWhyAllToAllInPlaceBroken);
}

// AllToAll out-of-place still hits the "AllReduce only" rule today - this is
// not a positive case for the in-place rule, but pins that in-place is
// specifically what that rule refuses, not AllToAll itself.
TEST(
    ValidityTableTest,
    validate_with_alltoall_and_out_of_place_then_hits_unsupported_collective_rule) {
    CollectiveParams params(Collective::ONECCL_ALLTOALL, onecclFloat32,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    params.is_in_place = false;
    LaunchTopology topology;

    std::string why;
    EXPECT_FALSE(
        ValidityTable::is_valid(params, all_capable(), topology, &why));
    EXPECT_NE(why, kWhyAllToAllInPlaceBroken);
}

// Every collective other than AllReduce is refused, "AllReduce only".
TEST(ValidityTableTest, validate_with_non_allreduce_collective_then_refuses) {
    LaunchTopology topology;

    for (Collective collective : kAllCollectives) {
        if (collective == Collective::ONECCL_ALLREDUCE) {
            continue;
        }
        CollectiveParams params(collective, onecclFloat32,
                                PatternOp::ONECCL_SUM,
                                /*count=*/4);
        std::string why;
        EXPECT_FALSE(
            ValidityTable::is_valid(params, all_capable(), topology, &why))
            << "collective=" << static_cast<int>(collective);
        EXPECT_EQ(why, kWhyAllReduceOnly)
            << "collective=" << static_cast<int>(collective);
    }
}

TEST(ValidityTableTest,
     validate_with_allreduce_and_no_other_refusal_then_allows) {
    CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclFloat32,
                            PatternOp::ONECCL_SUM, /*count=*/4);
    LaunchTopology topology;

    std::string why;
    EXPECT_TRUE(ValidityTable::is_valid(params, all_capable(), topology, &why))
        << why;
}

// ParamExpander::expand's refusals out-param (allreduce_test.cpp's
// expand_allreduce() passes no refusals argument).
TEST(ParamExpanderTest,
     expand_with_refusals_out_param_then_records_every_refused_row) {
    DeviceCapabilities capabilities = all_capable();
    capabilities.supports_fp64 = false; // makes every fp64 row a refusal
    LaunchTopology topology;

    const std::vector<onecclDataType_t> dtypes = {onecclFloat32, onecclFloat64};
    const std::vector<PatternOp> ops = {PatternOp::ONECCL_SUM};
    const std::vector<size_t> counts = {4};

    std::vector<ParamExpander::Refusal> refusals;
    const std::vector<CollectiveParams> accepted =
        ParamExpander::expand(Collective::ONECCL_ALLREDUCE, dtypes, ops, counts,
                              capabilities, topology, &refusals);

    // fp32 is accepted, fp64 is refused for lacking device support.
    ASSERT_EQ(accepted.size(), 1u);
    EXPECT_EQ(accepted[0].dtype, onecclFloat32);

    ASSERT_EQ(refusals.size(), 1u);
    EXPECT_EQ(refusals[0].params.dtype, onecclFloat64);
    EXPECT_EQ(refusals[0].why, kWhyFp64Unsupported);
}

// The refusals vector is documented to only ever be appended to, so a caller
// can accumulate refusals for several collectives' matrices in one vector.
TEST(ParamExpanderTest,
     expand_with_refusals_out_param_and_repeated_calls_then_accumulates) {
    DeviceCapabilities capabilities = all_capable();
    LaunchTopology topology;

    const std::vector<onecclDataType_t> dtypes = {onecclFloat32};
    const std::vector<PatternOp> ops = {PatternOp::ONECCL_SUM};
    const std::vector<size_t> counts = {4};

    std::vector<ParamExpander::Refusal> refusals;
    // AllReduce accepts this row; Broadcast refuses it ("AllReduce only").
    ParamExpander::expand(Collective::ONECCL_ALLREDUCE, dtypes, ops, counts,
                          capabilities, topology, &refusals);
    EXPECT_TRUE(refusals.empty());

    ParamExpander::expand(Collective::ONECCL_BROADCAST, dtypes, ops, counts,
                          capabilities, topology, &refusals);
    ASSERT_EQ(refusals.size(), 1u);
    EXPECT_EQ(refusals[0].why, kWhyAllReduceOnly);
}

// expand() judges a row in place, so a rule that refuses in place records the
// refusal.
TEST(ParamExpanderTest, expand_with_in_place_flag_then_validity_rules_see_it) {
    LaunchTopology topology;
    const std::vector<onecclDataType_t> dtypes = {onecclFloat32};
    const std::vector<PatternOp> ops = {PatternOp::ONECCL_SUM};
    const std::vector<size_t> counts = {4};

    std::vector<ParamExpander::Refusal> refusals;
    const std::vector<CollectiveParams> accepted = ParamExpander::expand(
        Collective::ONECCL_ALLTOALL, dtypes, ops, counts, all_capable(),
        topology, &refusals, /*is_in_place=*/true);

    EXPECT_TRUE(accepted.empty());
    ASSERT_EQ(refusals.size(), 1u);
    EXPECT_EQ(refusals[0].why, kWhyAllToAllInPlaceBroken);
    EXPECT_TRUE(refusals[0].params.is_in_place);
}
