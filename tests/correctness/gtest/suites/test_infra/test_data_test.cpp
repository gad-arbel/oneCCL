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

#include "data/bf16.h"
#include "data/data_spec.h"
#include "data/host_buffer.h"
#include "data/patterns/data_pattern.h"
#include "data/source_map.h"
#include "data/test_data.h"
#include "params/collective_params.h"
#include "params/param_expander.h"
#include "params/validity_table.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace oneccl_test;

// Basic smoke test for ITestData creation
// Pseudo code:
//   spec = int32 sum, 4 ranks, my_rank 0
//   data = ITestData::create(spec)
//   assert data != null
TEST(TestDataTest, create_with_valid_spec_then_returns_non_null) {
    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 10;
    spec.send_element_count = 10;
    spec.recv_element_count = 10;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr) << "Failed to create ITestData";
}

// Test fill_send
// Pseudo code:
//   data = create(int32 sum, my_rank 1, 4 ranks)
//   data.fill_send(buf)
//   assert every element == my_rank + buffer_index + 1 == 2
TEST(TestDataTest,
     fill_send_with_rank_constant_pattern_then_writes_rank_plus_one) {
    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 1;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    HostBuffer send_buffer(4 * sizeof(int32_t));
    test_data->fill_send(send_buffer);

    // RankConstantPattern: value = rank + buffer_index + 1 = 1 + 0 + 1 = 2
    const int32_t *values = send_buffer.as<int32_t>();
    for (size_t index = 0; index < 4; ++index) {
        EXPECT_EQ(values[index], 2) << "Element " << index << " incorrect";
    }
}

// Test poison_recv and assert_still_poisoned
// Pseudo code:
//   data.poison_recv(buf)
//   assert every byte == kPoisonByte
//   data.assert_still_poisoned(buf) passes
//   buf[5] = 0
//   assert assert_still_poisoned raises exactly 1 failure naming "byte 5"
TEST(TestDataTest,
     poison_recv_with_written_byte_then_assert_still_poisoned_reports_it) {
    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    HostBuffer recv_buffer(4 * sizeof(int32_t));
    test_data->poison_recv(recv_buffer);

    // All bytes should be 0xA5
    const std::byte *buffer_bytes = recv_buffer.data();
    for (size_t index = 0; index < recv_buffer.size(); ++index) {
        EXPECT_EQ(buffer_bytes[index], ITestData::kPoisonByte)
            << "Byte " << index << " not poisoned";
    }

    // assert_still_poisoned should pass
    test_data->assert_still_poisoned(recv_buffer);

    // One written byte must be reported, so a no-op check cannot pass
    recv_buffer.data()[5] = std::byte{0};
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::
                INTERCEPT_ONLY_CURRENT_THREAD,
            &failures);
        test_data->assert_still_poisoned(recv_buffer);
    }
    ASSERT_EQ(failures.size(), 1);
    EXPECT_NE(
        std::string(failures.GetTestPartResult(0).message()).find("byte 5"),
        std::string::npos)
        << failures.GetTestPartResult(0).message();
}

// Test compare with correct values
// Pseudo code:
//   buf = [10, 10, 10, 10]          // 1+2+3+4
//   mismatches = data.compare(buf, all_ranks, offset 0)
//   assert mismatches == 0 and report is empty
TEST(TestDataTest, compare_with_correct_values_then_reports_no_mismatch) {
    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    // Build expected: sum of (1, 2, 3, 4) = 10 for each element
    HostBuffer recv_buffer(4 * sizeof(int32_t));
    int32_t *values = recv_buffer.as<int32_t>();
    for (size_t index = 0; index < 4; ++index) {
        values[index] = 10; // Expected sum: 1+2+3+4 = 10
    }

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;
    source_map.source_base = 0; // Base index for the source data

    MismatchReport report;
    size_t mismatch_count =
        test_data->compare(recv_buffer, source_map, 0, report);
    EXPECT_EQ(mismatch_count, 0u) << "Should have no mismatches";
    EXPECT_TRUE(report.no_mismatch_recorded())
        << "Report should record no mismatch";
}

// Test compare refuses buffers that run past recv_element_count or end in a
// partial element
// Pseudo code:
//   compare(buf of recv_count + 1 elements) -> throws logic_error
//   compare(buf of recv_count elements + 1 stray byte) -> throws logic_error
TEST(TestDataTest, compare_with_buffer_outside_recv_range_then_throws) {
    constexpr size_t kElementCount = 4;
    constexpr int32_t kExpectedSum = 10;

    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = kElementCount;
    spec.send_element_count = kElementCount;
    spec.recv_element_count = kElementCount;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;

    // Five correct values against a four-element receive range
    HostBuffer oversize_buffer((kElementCount + 1) * sizeof(int32_t));
    int32_t *values = oversize_buffer.as<int32_t>();
    for (size_t index = 0; index < kElementCount + 1; ++index) {
        values[index] = kExpectedSum;
    }
    MismatchReport oversize_report;
    EXPECT_THROW(
        test_data->compare(oversize_buffer, source_map, 0, oversize_report),
        std::logic_error);

    // A trailing partial element
    HostBuffer partial_buffer(kElementCount * sizeof(int32_t) + 1);
    MismatchReport partial_report;
    EXPECT_THROW(
        test_data->compare(partial_buffer, source_map, 0, partial_report),
        std::logic_error);
}

// Test compare refuses variable segments whose source range lies outside the
// send buffer
// Pseudo code:
//   source_map = rank-per-segment, source_offsets {1000, 2000}
//   send buffer has only 2 elements
//   compare(...) -> throws logic_error
TEST(TestDataTest,
     compare_with_source_segment_outside_send_buffer_then_throws) {
    constexpr size_t kSendElementCount = 2;

    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = kSendElementCount;
    spec.send_element_count = kSendElementCount;
    spec.recv_element_count = 2 * kSendElementCount;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_RANK_PER_SEGMENT;
    source_map.segment_offsets = {0, kSendElementCount, 2 * kSendElementCount};
    source_map.source_offsets = {1000, 2000};

    HostBuffer recv_buffer(spec.recv_element_count * sizeof(int32_t));
    MismatchReport report;
    EXPECT_THROW(test_data->compare(recv_buffer, source_map, 0, report),
                 std::logic_error);
}

// Test compare rejects every single-element corruption in turn
// Pseudo code:
//   buf = [10, 10, 10, 10]
//   assert compare(buf) == 0
//   for i in 0..3:
//       buf[i] = 42
//       assert compare(buf) == 1 and report.count == 1
//       buf[i] = 10
TEST(TestDataTest,
     compare_with_single_element_corruption_then_reports_one_mismatch) {
    constexpr size_t kElementCount = 4;
    // RankConstantPattern over four ranks: 1 + 2 + 3 + 4
    constexpr int32_t kExpectedSum = 10;
    constexpr int32_t kCorruptedValue = 42;

    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = kElementCount;
    spec.send_element_count = kElementCount;
    spec.recv_element_count = kElementCount;
    spec.my_rank = 2;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;
    source_map.source_base = 0;

    // A correct receive buffer is accepted
    HostBuffer recv_buffer(kElementCount * sizeof(int32_t));
    int32_t *values = recv_buffer.as<int32_t>();
    for (size_t index = 0; index < kElementCount; ++index) {
        values[index] = kExpectedSum;
    }

    MismatchReport clean_report;
    EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, clean_report), 0u)
        << "Correct receive buffer should be accepted";

    // Then each single-element corruption in turn is rejected, and only that
    // element
    for (size_t corrupt_index = 0; corrupt_index < kElementCount;
         ++corrupt_index) {
        values[corrupt_index] = kCorruptedValue;

        MismatchReport report;
        EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, report), 1u)
            << "Corruption at element " << corrupt_index
            << " should be rejected";
        EXPECT_EQ(report.mismatch_count(), 1u)
            << "Report should hold exactly one mismatch for element "
            << corrupt_index;

        values[corrupt_index] =
            kExpectedSum; // restore before the next corruption
    }
}

// Test the failure text locates the corruption. MismatchReport
// exposes no string, so the text is read back through gtest's own interceptor.
// Pseudo code:
//   buf = [10, 42, 10, 10]           // corrupt index 1, my_rank 2
//   data.verify(buf)                  // intercept the gtest failure
//   assert exactly 1 failure
//   assert text contains "Rank 2", "[1]", "actual=42", "expected=10"
TEST(
    TestDataTest,
    verify_with_corrupted_element_then_failure_names_rank_index_actual_and_expected) {
    constexpr size_t kElementCount = 4;
    constexpr size_t kCorruptIndex = 1;
    constexpr int kMyRank = 2;
    constexpr int32_t kExpectedSum = 10;
    constexpr int32_t kCorruptedValue = 42;

    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = kElementCount;
    spec.send_element_count = kElementCount;
    spec.recv_element_count = kElementCount;
    spec.my_rank = kMyRank;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;
    source_map.source_base = 0;

    HostBuffer recv_buffer(kElementCount * sizeof(int32_t));
    int32_t *values = recv_buffer.as<int32_t>();
    for (size_t index = 0; index < kElementCount; ++index) {
        values[index] = kExpectedSum;
    }
    values[kCorruptIndex] = kCorruptedValue;

    // verify() is the gtest-facing wrapper: it raises the non-fatal failure
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::
                INTERCEPT_ONLY_CURRENT_THREAD,
            &failures);
        test_data->verify(recv_buffer, source_map, 0);
    }
    ASSERT_EQ(failures.size(), 1)
        << "verify() should raise exactly one failure";

    const std::string message = failures.GetTestPartResult(0).message();
    EXPECT_NE(message.find("Rank " + std::to_string(kMyRank)),
              std::string::npos)
        << "Failure text should name the rank: " << message;
    EXPECT_NE(message.find("[" + std::to_string(kCorruptIndex) + "]"),
              std::string::npos)
        << "Failure text should name the index: " << message;
    EXPECT_NE(message.find("actual=" + std::to_string(kCorruptedValue)),
              std::string::npos)
        << "Failure text should name the actual value: " << message;
    EXPECT_NE(message.find("expected=" + std::to_string(kExpectedSum)),
              std::string::npos)
        << "Failure text should name the expected value: " << message;
}

// Test is_supported
// Pseudo code:
//   int32 sum, 4 ranks          -> supported
//   int32 avg                    -> refused
//   uint8 max                    -> refused ("datatype"), create() == null
//   int32 sum, 46340 / 46341 ranks -> supported / refused ("overflow")
//   int32 prod, 9 / 10 ranks       -> supported / refused ("overflow")
//   fp32 prod, 2 ranks (bit-exact) -> supported
//   bf16 prod 6 ranks, fp32 avg 2 ranks (tolerance) -> refused ("tolerance")
TEST(TestDataTest,
     is_supported_with_dtype_op_and_rank_count_then_accepts_or_refuses) {
    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 10;
    spec.send_element_count = 10;
    spec.recv_element_count = 10;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1, 2, 3};
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    std::string why;
    bool spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_TRUE(spec_is_supported) << "Spec should be supported: " << why;

    // Integer avg is not supported
    spec.op = PatternOp::ONECCL_AVG;
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported) << "Integer avg should not be supported";

    // A dtype create() does not instantiate is refused, even when its
    // pattern, order and op are all valid
    spec.dtype = onecclUint8;
    spec.op = PatternOp::ONECCL_MAX;
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported) << "uint8 should not be supported";
    EXPECT_NE(why.find("datatype"), std::string::npos)
        << "why should mention the datatype: " << why;
    EXPECT_EQ(ITestData::create(spec), nullptr)
        << "create() should agree that uint8 is not instantiated";

    // Sum overflow: the conservative bound is rank_count * max_absolute_value
    // = n * n for RankConstant, which first exceeds int32 max at n=46341 -
    // is_supported() must refuse using the per-dtype bound.
    constexpr int kLastSumRankCount = 46340;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.comm_ranks.clear();
    for (int rank = 0; rank < kLastSumRankCount; ++rank) {
        spec.comm_ranks.push_back(rank);
    }
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_TRUE(spec_is_supported)
        << "int32 sum at the last fitting rank count: " << why;
    spec.comm_ranks.push_back(kLastSumRankCount);
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported)
        << "int32 sum overflow should not be supported";
    EXPECT_NE(why.find("overflow"), std::string::npos)
        << "why should mention overflow: " << why;

    // Prod overflow: RankConstant's max_absolute_value is rank_count, so the
    // conservative n^n bound first exceeds int32 range at n=10 (9^9 fits).
    constexpr int kLastProdRankCount = 9;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_PROD;
    spec.comm_ranks.clear();
    for (int rank = 0; rank < kLastProdRankCount; ++rank) {
        spec.comm_ranks.push_back(rank);
    }
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_TRUE(spec_is_supported)
        << "int32 prod at the last fitting rank count: " << why;
    spec.comm_ranks.push_back(kLastProdRankCount);
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported)
        << "int32 prod overflow should not be supported";
    EXPECT_NE(why.find("overflow"), std::string::npos)
        << "why should mention overflow: " << why;

    // A float prod whose exact product fits is bit-exact and supported:
    // fp32 prod at world_size 2 is 1 * 2
    spec.dtype = onecclFloat32;
    spec.op = PatternOp::ONECCL_PROD;
    spec.comm_ranks = {0, 1};
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_BIT_EXACT);
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_TRUE(spec_is_supported) << "fp32 prod at 2 ranks: " << why;

    // tolerance_bound() has only the sum-shaped window, so prod and avg
    // tolerance rows are refused: bf16 prod at 6 ranks (6! = 720 > 256) and
    // fp32 avg at 2 ranks
    spec.dtype = onecclBfloat16;
    spec.comm_ranks = {0, 1, 2, 3, 4, 5};
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported)
        << "prod tolerance rows should not be supported yet";
    EXPECT_NE(why.find("tolerance"), std::string::npos)
        << "why should mention tolerance: " << why;

    spec.dtype = onecclFloat32;
    spec.op = PatternOp::ONECCL_AVG;
    spec.comm_ranks = {0, 1};
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
    spec_is_supported = ITestData::is_supported(spec, &why);
    EXPECT_FALSE(spec_is_supported)
        << "avg tolerance rows should not be supported yet";
    EXPECT_NE(why.find("tolerance"), std::string::npos)
        << "why should mention tolerance: " << why;
}

// Test compare_mode over the M0 dtype x op matrix: every
// {fp32, bf16, int32} x {sum, max} row at world_size = 2 is bit-exact, with no
// hardware model consulted.
// Pseudo code:
//   for dtype in {fp32, bf16, int32}, op in {sum, max}:
//       assert compare_mode(spec, 2 ranks) == BIT_EXACT
TEST(
    TestDataTest,
    compare_mode_with_fp32_bf16_int32_sum_and_max_at_two_ranks_then_bit_exact) {
    struct ModeRow {
        onecclDataType_t dtype;
        const char *dtype_name;
        PatternOp op;
        const char *op_name;
    };
    static const ModeRow kModeRows[] = {
        {onecclFloat32, "fp32", PatternOp::ONECCL_SUM, "sum"},
        {onecclFloat32, "fp32", PatternOp::ONECCL_MAX, "max"},
        {onecclBfloat16, "bf16", PatternOp::ONECCL_SUM, "sum"},
        {onecclBfloat16, "bf16", PatternOp::ONECCL_MAX, "max"},
        {onecclInt32, "int32", PatternOp::ONECCL_SUM, "sum"},
        {onecclInt32, "int32", PatternOp::ONECCL_MAX, "max"},
    };

    DataSpec spec;
    spec.dtype = onecclInt32;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.comm_ranks = {0, 1}; // world_size = 2
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    for (const ModeRow &row : kModeRows) {
        spec.dtype = row.dtype;
        spec.op = row.op;
        EXPECT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_BIT_EXACT)
            << row.dtype_name << " x " << row.op_name
            << " must be bit-exact at world_size 2";
    }
}

// Test the bf16 sum boundary is exactly the closed-form rule:
// RankConstantPattern's exact sum is n(n+1)/2, bf16
// represents integers exactly up to 256, so the mode is bit-exact while
// n(n+1)/2 <= 256 -- n = 22 (253) is the last such rank count, n = 23 (276) the
// first that needs tolerance.
// Pseudo code:
//   for n in 1..24 ranks:
//       exact = n(n+1)/2
//       assert compare_mode(bf16 sum, n) == (exact <= 256 ? BIT_EXACT :
//       TOLERANCE)
TEST(
    TestDataTest,
    compare_mode_with_bf16_sum_and_rank_count_then_switches_to_tolerance_past_256) {
    constexpr size_t kBf16ExactIntegerRange = 256;
    constexpr size_t kMaxRankCount = 24;

    DataSpec spec;
    spec.dtype = onecclBfloat16;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    for (size_t rank_count = 1; rank_count <= kMaxRankCount; ++rank_count) {
        spec.comm_ranks.clear();
        for (size_t position = 0; position < rank_count; ++position) {
            spec.comm_ranks.push_back(static_cast<int>(position));
        }

        const size_t exact_sum = rank_count * (rank_count + 1) / 2;
        const CompareMode expected_mode = (exact_sum <= kBf16ExactIntegerRange)
                                              ? CompareMode::ONECCL_BIT_EXACT
                                              : CompareMode::ONECCL_TOLERANCE;

        EXPECT_EQ(ITestData::compare_mode(spec), expected_mode)
            << "bf16 sum at rank_count=" << rank_count
            << ": exact sum n(n+1)/2=" << exact_sum
            << " against bf16's exact-integer range " << kBf16ExactIntegerRange;
    }
}

// Test tolerance_bound against the closed form
// gamma(n-1) * absolute_sum + u * |expected|, gamma(k) = k*u / (1 - k*u).
// Pseudo code:
//   bf16, 1 rank   -> u * |expected|
//   bf16, 8 ranks  -> gamma(7) * abs_sum + u * |expected|
//   int32          -> 0
//   bf16, 128 ranks -> bound < |expected|   // a zeroed output is still caught
TEST(TestDataTest,
     tolerance_bound_with_dtype_and_rank_count_then_matches_closed_form) {
    constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
    constexpr double kAbsoluteSum = 1000.0;
    constexpr double kExpected = 1000.0;

    // One rank is a copy: only the reference's own cast is charged.
    EXPECT_DOUBLE_EQ(
        ITestData::tolerance_bound(onecclBfloat16, 1, kAbsoluteSum, kExpected),
        kBf16UnitRoundoff * kExpected);

    // bf16 at 8 ranks: gamma(7) = 7u / (1 - 7u).
    const double gamma_7 =
        7.0 * kBf16UnitRoundoff / (1.0 - 7.0 * kBf16UnitRoundoff);
    EXPECT_DOUBLE_EQ(
        ITestData::tolerance_bound(onecclBfloat16, 8, kAbsoluteSum, kExpected),
        gamma_7 * kAbsoluteSum + kBf16UnitRoundoff * kExpected);

    // Integers are bit-exact: no window at any rank count.
    EXPECT_EQ(
        ITestData::tolerance_bound(onecclInt32, 8, kAbsoluteSum, kExpected),
        0.0);

    // At the last rank count is_supported() accepts for bf16 (128), the window
    // still excludes a zeroed output: bound < |expected| when every
    // contribution is positive (absolute_sum == expected).
    EXPECT_LT(ITestData::tolerance_bound(onecclBfloat16, 128, kAbsoluteSum,
                                         kExpected),
              kExpected);
}

// Test the too-wide-window refusal: a tolerance row with (n-1)*u >= 1/2 is
// refused, so bf16 sum is supported at 128 ranks and refused at 129.
// Pseudo code:
//   bf16 sum, 128 ranks (tolerance) -> supported
//   bf16 sum, 129 ranks             -> refused ("too wide")
TEST(TestDataTest,
     is_supported_with_bf16_sum_at_129_ranks_then_refuses_too_wide_tolerance) {
    DataSpec spec;
    spec.dtype = onecclBfloat16;
    spec.op = PatternOp::ONECCL_SUM;
    spec.count = 4;
    spec.send_element_count = 4;
    spec.recv_element_count = 4;
    spec.my_rank = 0;
    spec.buffer_index = 0;
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;

    for (int rank = 0; rank < 128; ++rank) {
        spec.comm_ranks.push_back(rank);
    }
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
    std::string why;
    EXPECT_TRUE(ITestData::is_supported(spec, &why))
        << "bf16 sum at 128 ranks should be supported: " << why;

    spec.comm_ranks.push_back(128);
    EXPECT_FALSE(ITestData::is_supported(spec, &why))
        << "bf16 sum at 129 ranks has a too-wide tolerance window";
    EXPECT_NE(why.find("too wide"), std::string::npos)
        << "why should mention the too-wide window: " << why;
}

// Build a PreMulSum spec over ranks 0..rank_count-1 through
// CollectiveParams::to_data_spec(), so the scalars are the ones a real row gets
// (rank + 1).
static DataSpec make_pre_mul_sum_spec(onecclDataType_t dtype,
                                      size_t rank_count) {
    std::vector<int> comm_ranks;
    for (size_t position = 0; position < rank_count; ++position) {
        comm_ranks.push_back(static_cast<int>(position));
    }
    const CollectiveParams params(Collective::ONECCL_ALLREDUCE, dtype,
                                  PatternOp::ONECCL_PRE_MUL_SUM, 4);
    return params.to_data_spec(0, comm_ranks, 0);
}

// Test PreMulSum's bit-exact boundary uses the scaled sum sum((r+1)^2), not
// n(n+1)/2: bf16 is exact to 256, so 8 ranks (204) are bit-exact and 9 ranks
// (285) need tolerance, though the unscaled sum (45) would still fit.
// Pseudo code:
//   scalars = rank + 1, so exact sum = sum((r+1)^2)
//   bf16 pre_mul_sum, 8 ranks (204) -> BIT_EXACT
//   bf16 pre_mul_sum, 9 ranks (285) -> TOLERANCE
TEST(TestDataTest, compare_mode_with_pre_mul_sum_then_uses_scaled_range) {
    EXPECT_EQ(ITestData::compare_mode(make_pre_mul_sum_spec(onecclBfloat16, 8)),
              CompareMode::ONECCL_BIT_EXACT);
    EXPECT_EQ(ITestData::compare_mode(make_pre_mul_sum_spec(onecclBfloat16, 9)),
              CompareMode::ONECCL_TOLERANCE);
}

// Test PreMulSum's overflow refusal includes the largest scalar: the bound is
// n * (n * n) = n^3, which first exceeds int32 max (2147483647) at n = 1291,
// though the unscaled bound n * n would still fit; n = 1290 still fits.
// Pseudo code:
//   bound = n^3
//   int32 pre_mul_sum, 1290 ranks -> supported
//   int32 pre_mul_sum, 1291 ranks -> refused ("overflow")
TEST(TestDataTest,
     is_supported_with_pre_mul_sum_overflow_then_uses_scaled_range) {
    std::string why;
    EXPECT_TRUE(
        ITestData::is_supported(make_pre_mul_sum_spec(onecclInt32, 1290), &why))
        << "int32 pre_mul_sum at 1290 ranks should be supported: " << why;
    EXPECT_FALSE(
        ITestData::is_supported(make_pre_mul_sum_spec(onecclInt32, 1291), &why))
        << "int32 pre_mul_sum at 1291 ranks overflows once scaled";
    EXPECT_NE(why.find("overflow"), std::string::npos)
        << "why should mention overflow: " << why;
}

// Build a RankConstant / Sequential spec over the given ranks
static DataSpec make_spec(onecclDataType_t dtype, PatternOp op,
                          std::vector<int> comm_ranks, size_t count = 4) {
    DataSpec spec;
    spec.dtype = dtype;
    spec.op = op;
    spec.count = count;
    spec.send_element_count = count;
    spec.recv_element_count = count;
    spec.my_rank = comm_ranks.empty() ? 0 : comm_ranks.front();
    spec.buffer_index = 0;
    spec.comm_ranks = std::move(comm_ranks);
    spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
    spec.order = OrderKind::ONECCL_SEQUENTIAL;
    return spec;
}

// Ranks 0..rank_count-1
static std::vector<int> first_ranks(size_t rank_count) {
    std::vector<int> ranks;
    for (size_t position = 0; position < rank_count; ++position) {
        ranks.push_back(static_cast<int>(position));
    }
    return ranks;
}

// Test compare_mode and max_absolute_value read the actual comm_ranks, not
// 0..n-1: a subcommunicator {128, 129, 130} sums to 390, past bf16's 256.
// Pseudo code:
//   bf16 sum over ranks {128, 129, 130} (390 > 256) -> TOLERANCE
//   max_absolute_value(int32 sum over {3, 7}) == 8      // rank 7 + 1
TEST(TestDataTest, compare_mode_with_sparse_comm_ranks_then_uses_actual_ranks) {
    const DataSpec spec =
        make_spec(onecclBfloat16, PatternOp::ONECCL_SUM, {128, 129, 130});
    EXPECT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);

    const DataSpec sparse_spec =
        make_spec(onecclInt32, PatternOp::ONECCL_SUM, {3, 7});
    EXPECT_DOUBLE_EQ(IDataPattern::max_absolute_value(sparse_spec), 8.0)
        << "largest value is rank 7 + 1";
}

// Test fp16's exact range is 2048, not fp64's 2^53: 63 ranks sum to 2016
// (bit-exact), 64 ranks to 2080 (tolerance).
// Pseudo code:
//   fp16 sum, 63 ranks (2016 <= 2048) -> BIT_EXACT
//   fp16 sum, 64 ranks (2080 > 2048)  -> TOLERANCE
TEST(TestDataTest, compare_mode_with_fp16_sum_then_exact_range_ends_at_2048) {
    EXPECT_EQ(ITestData::compare_mode(make_spec(
                  onecclFloat16, PatternOp::ONECCL_SUM, first_ranks(63))),
              CompareMode::ONECCL_BIT_EXACT);
    EXPECT_EQ(ITestData::compare_mode(make_spec(
                  onecclFloat16, PatternOp::ONECCL_SUM, first_ranks(64))),
              CompareMode::ONECCL_TOLERANCE);
}

// Test buffer_index enters the exact sum: bf16 at 22 ranks is 253 for buffer
// 0 and 253 + 22 = 275 > 256 for buffer 1.
// Pseudo code:
//   bf16 sum, 22 ranks, buffer 0 (253) -> BIT_EXACT
//   same spec, buffer 1 (253 + 22 = 275) -> TOLERANCE
TEST(TestDataTest,
     compare_mode_with_nonzero_buffer_index_then_counts_it_in_exact_sum) {
    DataSpec spec =
        make_spec(onecclBfloat16, PatternOp::ONECCL_SUM, first_ranks(22));
    EXPECT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_BIT_EXACT);
    spec.buffer_index = 1;
    EXPECT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
}

// Test a fractional PreMulSum scalar is never assumed exact, even when the
// scaled sum is small.
// Pseudo code:
//   fp32 pre_mul_sum, 3 ranks            -> BIT_EXACT
//   scalars[1] = 0.1                       -> TOLERANCE
TEST(TestDataTest,
     compare_mode_with_pre_mul_sum_and_fractional_scalar_then_tolerance) {
    DataSpec spec = make_pre_mul_sum_spec(onecclFloat32, 3);
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_BIT_EXACT);
    spec.pre_mul_scalars[1] = 0.1;
    EXPECT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
}

// Test the PreMulSum reference casts the send value and the scalar to T before
// it multiplies them, as the product does. A product of the raw doubles lands
// outside the window in both rows below.
// Pseudo code:
//   bf16, ranks {2}, scalar 1.003: typed 3 * 1     -> expect 3
//   bf16, ranks {256}, factory scalar 257: typed 256 * 256 -> expect 65536
//   for each row: buf = [expected]; assert compare(buf) == 0
TEST(TestDataTest, compare_with_pre_mul_sum_then_casts_operands_to_dtype) {
    struct PreMulSumRow {
        DataSpec spec;
        double expected;
    };
    DataSpec fractional_spec =
        make_spec(onecclBfloat16, PatternOp::ONECCL_PRE_MUL_SUM, {2}, 1);
    fractional_spec.pre_mul_scalars = {1.003};
    const CollectiveParams params(Collective::ONECCL_ALLREDUCE, onecclBfloat16,
                                  PatternOp::ONECCL_PRE_MUL_SUM, 1);
    const PreMulSumRow kRows[] = {
        {fractional_spec, 3.0},
        {params.to_data_spec(256, {256}, 0), 65536.0},
    };

    for (const PreMulSumRow &row : kRows) {
        auto test_data = ITestData::create(row.spec);
        ASSERT_NE(test_data, nullptr);

        SourceMap source_map;
        source_map.contributors = Contributors::ONECCL_ALL_RANKS;

        HostBuffer recv_buffer(sizeof(Bf16));
        recv_buffer.as<Bf16>()[0] = Bf16::from_double(row.expected);
        MismatchReport report;
        EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, report), 0u)
            << "scalar=" << row.spec.pre_mul_scalars[0]
            << " expected=" << row.expected;
    }
}

// Test compare() crosses the scratch-window boundary: the second window is
// checked, read at the right offset, and reported at its global index. The
// last window holds one element, so it is also a partial window.
// Pseudo code:
//   int32 sum, 4 ranks, kWindowElementCount + 1 elements, all 10
//   assert compare(buf) == 0
//   buf[kWindowElementCount] = 42
//   verify(buf) -> exactly 1 failure naming "[kWindowElementCount]"
TEST(TestDataTest,
     compare_with_corruption_past_window_boundary_then_reports_global_index) {
    constexpr size_t kElementCount = ITestData::kWindowElementCount + 1;
    constexpr size_t kCorruptIndex = ITestData::kWindowElementCount;
    const DataSpec spec = make_spec(onecclInt32, PatternOp::ONECCL_SUM,
                                    first_ranks(4), kElementCount);
    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;

    HostBuffer recv_buffer(kElementCount * sizeof(int32_t));
    int32_t *values = recv_buffer.as<int32_t>();
    for (size_t index = 0; index < kElementCount; ++index) {
        values[index] = 10;
    }
    MismatchReport clean_report;
    EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, clean_report), 0u);

    values[kCorruptIndex] = 42;
    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::
                INTERCEPT_ONLY_CURRENT_THREAD,
            &failures);
        test_data->verify(recv_buffer, source_map, 0);
    }
    ASSERT_EQ(failures.size(), 1)
        << "verify() should raise exactly one failure";
    const std::string message = failures.GetTestPartResult(0).message();
    EXPECT_NE(message.find("[" + std::to_string(kCorruptIndex) + "]"),
              std::string::npos)
        << "Failure text should name the global index: " << message;
}

// Test expand() sizes the placeholder communicator from the topology: bf16
// sum survives at 128 ranks (2 x 64) and is refused at 129 (43 x 3).
// Pseudo code:
//   caps = bf16 supported
//   expand(bf16 sum, topology 2 x 64)  -> 1 row, no refusal
//   expand(bf16 sum, topology 43 x 3)  -> 0 rows, 1 refusal ("too wide")
TEST(ParamExpanderTest,
     expand_with_bf16_sum_and_topology_then_sizes_placeholder_communicator) {
    DeviceCapabilities device_capabilities;
    device_capabilities.supports_bf16 = true;
    LaunchTopology topology;
    std::vector<ParamExpander::Refusal> refusals;

    topology.ranks_per_node = 2;
    topology.nodes = 64;
    EXPECT_EQ(ParamExpander::expand(Collective::ONECCL_ALLREDUCE,
                                    {onecclBfloat16}, {PatternOp::ONECCL_SUM},
                                    {4}, device_capabilities, topology,
                                    &refusals)
                  .size(),
              1u);
    EXPECT_TRUE(refusals.empty());

    topology.ranks_per_node = 43;
    topology.nodes = 3;
    EXPECT_TRUE(ParamExpander::expand(Collective::ONECCL_ALLREDUCE,
                                      {onecclBfloat16}, {PatternOp::ONECCL_SUM},
                                      {4}, device_capabilities, topology,
                                      &refusals)
                    .empty());
    ASSERT_EQ(refusals.size(), 1u);
    EXPECT_NE(refusals[0].why.find("too wide"), std::string::npos)
        << refusals[0].why;
}

// Test expand() refuses rows the device cannot run, and appends to the
// refusal list without clearing earlier entries.
// Pseudo code:
//   caps = nothing supported
//   expand(bf16 sum)  -> 0 rows, refusals = ["bf16 unsupported"]
//   expand(fp64 sum)  -> 0 rows, refusals = ["bf16 ...", "fp64 unsupported"]
TEST(ParamExpanderTest,
     expand_with_unsupported_device_then_refuses_and_appends_refusals) {
    const DeviceCapabilities device_capabilities;
    const LaunchTopology topology;
    std::vector<ParamExpander::Refusal> refusals;

    EXPECT_TRUE(ParamExpander::expand(Collective::ONECCL_ALLREDUCE,
                                      {onecclBfloat16}, {PatternOp::ONECCL_SUM},
                                      {4}, device_capabilities, topology,
                                      &refusals)
                    .empty());
    EXPECT_TRUE(ParamExpander::expand(Collective::ONECCL_ALLREDUCE,
                                      {onecclFloat64}, {PatternOp::ONECCL_SUM},
                                      {4}, device_capabilities, topology,
                                      &refusals)
                    .empty());
    ASSERT_EQ(refusals.size(), 2u) << "the second call must append";
    EXPECT_EQ(refusals[0].why, kWhyBf16Unsupported);
    EXPECT_EQ(refusals[1].why, kWhyFp64Unsupported);
}

// Test the tolerance path end to end: bf16 sum at 64 ranks has exact sum
// 2080. One ULP off (2096) is inside the window, a zeroed element and a NaN
// are not.
// Pseudo code:
//   bf16 sum, 64 ranks, exact sum 2080 (tolerance mode)
//   buf = [2080, 2096, 2080, 2080]      // 2096 is one ULP off
//   assert compare(buf) == 0
//   buf[2] = 0; buf[3] = NaN
//   assert compare(buf) == 2
TEST(
    TestDataTest,
    compare_with_bf16_tolerance_then_accepts_one_ulp_and_rejects_zero_and_nan) {
    constexpr size_t kElementCount = 4;
    constexpr double kExactSum = 2080.0;
    constexpr double kOneUlpOff = 2096.0;
    const DataSpec spec = make_spec(onecclBfloat16, PatternOp::ONECCL_SUM,
                                    first_ranks(64), kElementCount);
    ASSERT_EQ(ITestData::compare_mode(spec), CompareMode::ONECCL_TOLERANCE);
    std::string why;
    ASSERT_TRUE(ITestData::is_supported(spec, &why)) << why;

    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;

    HostBuffer recv_buffer(kElementCount * sizeof(Bf16));
    Bf16 *values = recv_buffer.as<Bf16>();
    for (size_t index = 0; index < kElementCount; ++index) {
        values[index] = Bf16::from_double(kExactSum);
    }
    values[1] = Bf16::from_double(kOneUlpOff);
    MismatchReport in_window_report;
    EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, in_window_report),
              0u);

    values[2] = Bf16::from_double(0.0);
    values[3] = Bf16::from_float(std::numeric_limits<float>::quiet_NaN());
    MismatchReport out_of_window_report;
    EXPECT_EQ(
        test_data->compare(recv_buffer, source_map, 0, out_of_window_report),
        2u)
        << "a zeroed element and a NaN must both be mismatches";
}

// Test the min, max and prod folds over ranks {0, 1, 2, 3} (values 1..4)
// Pseudo code:
//   ranks 0..3 contribute 1..4
//   for (max, 4), (min, 1), (prod, 24):
//       buf = [expected x 4]; assert compare(buf) == 0
//       buf[0] += 1;          assert compare(buf) == 1
TEST(TestDataTest, compare_with_max_min_and_prod_then_checks_fold_result) {
    struct FoldRow {
        PatternOp op;
        const char *op_name;
        int32_t expected;
    };
    static const FoldRow kFoldRows[] = {
        {PatternOp::ONECCL_MAX, "max", 4},
        {PatternOp::ONECCL_MIN, "min", 1},
        {PatternOp::ONECCL_PROD, "prod", 24},
    };
    constexpr size_t kElementCount = 4;

    for (const FoldRow &row : kFoldRows) {
        const DataSpec spec = make_spec(onecclInt32, row.op, first_ranks(4));
        auto test_data = ITestData::create(spec);
        ASSERT_NE(test_data, nullptr) << row.op_name;

        SourceMap source_map;
        source_map.contributors = Contributors::ONECCL_ALL_RANKS;

        HostBuffer recv_buffer(kElementCount * sizeof(int32_t));
        int32_t *values = recv_buffer.as<int32_t>();
        for (size_t index = 0; index < kElementCount; ++index) {
            values[index] = row.expected;
        }
        MismatchReport clean_report;
        EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, clean_report),
                  0u)
            << row.op_name;

        values[0] = row.expected + 1;
        MismatchReport corrupt_report;
        EXPECT_EQ(
            test_data->compare(recv_buffer, source_map, 0, corrupt_report), 1u)
            << row.op_name;
    }
}

// Test the single-rank and uniform per-segment source shapes
// Pseudo code:
//   single_rank, root 2: buf = [3, 3, 3, 3]     -> 0 mismatches
//   rank_per_segment, 2 segments of 2:
//       buf = [1, 1, 2, 2]                       -> 0 mismatches
//       also set source_offsets (unused here)    -> throws logic_error
TEST(TestDataTest,
     compare_with_single_rank_and_rank_per_segment_then_matches_source) {
    // SINGLE_RANK: every element is root 2's value, 3
    {
        const DataSpec spec =
            make_spec(onecclInt32, PatternOp::ONECCL_SUM, first_ranks(4));
        auto test_data = ITestData::create(spec);
        ASSERT_NE(test_data, nullptr);

        SourceMap source_map;
        source_map.contributors = Contributors::ONECCL_SINGLE_RANK;
        source_map.root = 2;

        HostBuffer recv_buffer(4 * sizeof(int32_t));
        int32_t *values = recv_buffer.as<int32_t>();
        for (size_t index = 0; index < 4; ++index) {
            values[index] = 3;
        }
        MismatchReport report;
        EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, report), 0u);
    }

    // RANK_PER_SEGMENT uniform: two segments of two, from ranks 0 and 1
    {
        DataSpec spec =
            make_spec(onecclInt32, PatternOp::ONECCL_SUM, first_ranks(2), 2);
        spec.recv_element_count = 4;
        auto test_data = ITestData::create(spec);
        ASSERT_NE(test_data, nullptr);

        SourceMap source_map;
        source_map.contributors = Contributors::ONECCL_RANK_PER_SEGMENT;
        source_map.segment_element_count = 2;

        HostBuffer recv_buffer(4 * sizeof(int32_t));
        int32_t *values = recv_buffer.as<int32_t>();
        const int32_t kExpected[] = {1, 1, 2, 2};
        std::memcpy(values, kExpected, sizeof(kExpected));
        MismatchReport report;
        EXPECT_EQ(test_data->compare(recv_buffer, source_map, 0, report), 0u);

        // Fields the uniform shape does not use are refused, not ignored
        source_map.source_offsets = {100, 200};
        MismatchReport refused_report;
        EXPECT_THROW(
            test_data->compare(recv_buffer, source_map, 0, refused_report),
            std::logic_error);
    }
}

// Test chained compare() calls must continue where the last one ended
// Pseudo code:
//   tile = [10, 10, 10, 10] of an 8-element buffer
//   compare(tile, offset 0)  -> 0 mismatches
//   compare(tile, offset 5)  -> throws logic_error   // gap after element 4
//   compare(tile, offset 4)  -> 0 mismatches
TEST(TestDataTest, compare_with_offset_gap_then_throws) {
    constexpr size_t kElementCount = 8;
    constexpr size_t kTileElementCount = 4;
    const DataSpec spec = make_spec(onecclInt32, PatternOp::ONECCL_SUM,
                                    first_ranks(4), kElementCount);
    auto test_data = ITestData::create(spec);
    ASSERT_NE(test_data, nullptr);

    SourceMap source_map;
    source_map.contributors = Contributors::ONECCL_ALL_RANKS;

    HostBuffer tile(kTileElementCount * sizeof(int32_t));
    int32_t *values = tile.as<int32_t>();
    for (size_t index = 0; index < kTileElementCount; ++index) {
        values[index] = 10;
    }

    MismatchReport report;
    EXPECT_EQ(test_data->compare(tile, source_map, 0, report), 0u);
    EXPECT_THROW(
        test_data->compare(tile, source_map, kTileElementCount + 1, report),
        std::logic_error);
    EXPECT_EQ(test_data->compare(tile, source_map, kTileElementCount, report),
              0u);
}

// Test create() refuses a spec whose my_rank is not in comm_ranks
// Pseudo code:
//   spec.comm_ranks = {0, 1}; spec.my_rank = 5
//   create(spec) -> throws logic_error
TEST(TestDataTest, create_with_my_rank_not_in_comm_ranks_then_throws) {
    DataSpec spec = make_spec(onecclInt32, PatternOp::ONECCL_SUM, {0, 1});
    spec.my_rank = 5;
    EXPECT_THROW(ITestData::create(spec), std::logic_error);
}

// Test is_supported() and create() agree on every dtype: a supported spec
// always builds, a refused dtype never does
// Pseudo code:
//   for each dtype in all 10 dtypes:
//       assert is_supported(max, 2 ranks) == (create(spec) != null)
TEST(TestDataTest, is_supported_with_every_dtype_then_agrees_with_create) {
    static const onecclDataType_t kAllDtypes[] = {
        onecclInt8,    onecclUint8,   onecclInt32,   onecclUint32,
        onecclInt64,   onecclUint64,  onecclFloat16, onecclFloat32,
        onecclFloat64, onecclBfloat16};

    for (onecclDataType_t dtype : kAllDtypes) {
        const DataSpec spec =
            make_spec(dtype, PatternOp::ONECCL_MAX, first_ranks(2));
        std::string why;
        const bool spec_is_supported = ITestData::is_supported(spec, &why);
        EXPECT_EQ(spec_is_supported, ITestData::create(spec) != nullptr)
            << "dtype=" << static_cast<int>(dtype) << " why=" << why;
    }
}

// Test Bf16 rounds to nearest even and keeps NaN a NaN
// Pseudo code:
//   1.0f              -> 0x3F80
//   tie, even below   -> 0x3F80   (round down)
//   tie, even above   -> 0x3F82   (round up)
//   three NaN patterns -> still NaN after the round trip
TEST(Bf16Test, from_float_with_ties_and_nan_then_rounds_to_even_and_keeps_nan) {
    auto from_bits = [](uint32_t bit_pattern) {
        float value;
        std::memcpy(&value, &bit_pattern, sizeof(value));
        return Bf16::from_float(value).bits;
    };

    EXPECT_EQ(from_bits(0x3F800000u), 0x3F80u) << "1.0 is exact";
    EXPECT_EQ(from_bits(0x3F808000u), 0x3F80u) << "tie rounds to even (down)";
    EXPECT_EQ(from_bits(0x3F818000u), 0x3F82u) << "tie rounds to even (up)";

    for (uint32_t nan_bits : {0x7F800001u, 0x7FFFFFFFu, 0xFFC00000u}) {
        const Bf16 nan = Bf16{from_bits(nan_bits)};
        EXPECT_TRUE(std::isnan(nan.to_float()))
            << "0x" << std::hex << nan_bits << " -> 0x" << nan.bits;
    }
}

// Test MismatchReport's kMaxReported truncation still states the true total
// Pseudo code:
//   add kMaxReported + 3 mismatches
//   assert mismatch_count == kMaxReported + 3
//   fail_now_with_report(rank 0) -> 1 failure, text contains "and 3 more"
TEST(MismatchReportTest,
     fail_now_with_report_with_more_than_max_reported_then_states_true_total) {
    constexpr size_t kExtraMismatches = 3;
    const size_t total_mismatches =
        MismatchReport::kMaxReported + kExtraMismatches;

    MismatchReport report;
    for (size_t index = 0; index < total_mismatches; ++index) {
        report.add_single_mismatch(index, /*actual=*/1.0, /*expected=*/0.0,
                                   /*tolerance=*/0.0);
    }
    EXPECT_EQ(report.mismatch_count(), total_mismatches);

    ::testing::TestPartResultArray failures;
    {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::
                INTERCEPT_ONLY_CURRENT_THREAD,
            &failures);
        report.fail_now_with_report(/*rank=*/0);
    }
    ASSERT_EQ(failures.size(), 1);

    const std::string message = failures.GetTestPartResult(0).message();
    EXPECT_NE(message.find("and " + std::to_string(kExtraMismatches) + " more"),
              std::string::npos)
        << message;
}

TEST(TestDataTest,
     fill_send_with_counts_around_window_boundary_then_writes_rank_plus_one) {
    constexpr int32_t kExpectedValue =
        2; // RankConstantPattern: rank(1)+buffer_index(0)+1
    const std::vector<size_t> kBoundaryCounts = {
        ITestData::kWindowElementCount - 1, ITestData::kWindowElementCount,
        ITestData::kWindowElementCount + 1};

    for (size_t element_count : kBoundaryCounts) {
        DataSpec spec;
        spec.dtype = onecclInt32;
        spec.op = PatternOp::ONECCL_SUM;
        spec.count = element_count;
        spec.send_element_count = element_count;
        spec.recv_element_count = element_count;
        spec.my_rank = 1;
        spec.buffer_index = 0;
        spec.comm_ranks = {0, 1, 2, 3};
        spec.pattern = PatternKind::ONECCL_RANK_CONSTANT;
        spec.order = OrderKind::ONECCL_SEQUENTIAL;

        auto test_data = ITestData::create(spec);
        ASSERT_NE(test_data, nullptr) << "count=" << element_count;

        HostBuffer send_buffer(element_count * sizeof(int32_t));
        test_data->fill_send(send_buffer);

        const int32_t *values = send_buffer.as<int32_t>();
        for (size_t index = 0; index < element_count; ++index) {
            ASSERT_EQ(values[index], kExpectedValue)
                << "count=" << element_count << " index=" << index;
        }
    }
}
