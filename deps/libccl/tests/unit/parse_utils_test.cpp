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

// Unit tests for common/utils/parse_utils.hpp, the number syntax of the
// selection policy.

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

#include "common/utils/parse_utils.hpp"

using ccl::utils::parse_bool;
using ccl::utils::parse_double;
using ccl::utils::parse_int;
using ccl::utils::parse_number;
using ccl::utils::parse_size;

TEST(ParseSize, PlainDigits) {
    EXPECT_EQ(parse_size("0"), 0u);
    EXPECT_EQ(parse_size("12"), 12u);
    EXPECT_EQ(parse_size("18446744073709551615"), std::numeric_limits<std::uint64_t>::max());
}

TEST(ParseSize, DecimalSuffixes) {
    EXPECT_EQ(parse_size("12K"), 12000u);
    EXPECT_EQ(parse_size("12k"), 12000u);
    EXPECT_EQ(parse_size("12M"), 12000000u);
    EXPECT_EQ(parse_size("12m"), 12000000u);
    EXPECT_EQ(parse_size("12G"), 12000000000u);
    EXPECT_EQ(parse_size("12g"), 12000000000u);
    EXPECT_EQ(parse_size("2T"), 2000000000000u);
}

TEST(ParseSize, BinarySuffixes) {
    EXPECT_EQ(parse_size("12Kb"), 12u * 1024);
    EXPECT_EQ(parse_size("12kb"), 12u * 1024);
    EXPECT_EQ(parse_size("12KB"), 12u * 1024);
    EXPECT_EQ(parse_size("12Mb"), 12u * 1024 * 1024);
    EXPECT_EQ(parse_size("12mb"), 12u * 1024 * 1024);
    EXPECT_EQ(parse_size("12MB"), 12u * 1024 * 1024);
    EXPECT_EQ(parse_size("12Gb"), 12ull * 1024 * 1024 * 1024);
    EXPECT_EQ(parse_size("1Tb"), 1ull << 40);
    EXPECT_EQ(parse_size("0Mb"), 0u);
}

TEST(ParseSize, RejectsMalformed) {
    for (const char* s : { "",
                           "M",
                           "Mb",
                           "b",
                           "12b",
                           "12B",
                           "12MiB",
                           "12Mbb",
                           "12X",
                           "12 M",
                           " 12M",
                           "12M ",
                           "-1",
                           "+1",
                           "0x10",
                           "1.5M",
                           "1e3" }) {
        EXPECT_FALSE(parse_size(s).has_value()) << "'" << s << "'";
    }
}

TEST(ParseSize, RejectsOverflow) {
    EXPECT_FALSE(parse_size("18446744073709551616").has_value());
    EXPECT_FALSE(parse_size("18446744073709552K").has_value());
    EXPECT_FALSE(parse_size("17179869184Gb").has_value());
    EXPECT_TRUE(parse_size("17179869183Gb").has_value());
}

TEST(ParseInt, SignsAndRange) {
    EXPECT_EQ(parse_int("-1"), -1);
    EXPECT_EQ(parse_int("-2K"), -2000);
    EXPECT_EQ(parse_int("3Kb"), 3072);
    EXPECT_EQ(parse_int("-9223372036854775808"), std::numeric_limits<std::int64_t>::min());
    EXPECT_FALSE(parse_int("9223372036854775808").has_value());
    EXPECT_FALSE(parse_int("+1").has_value());
    EXPECT_FALSE(parse_int("-").has_value());
}

TEST(ParseDouble, FractionsAndSuffixes) {
    EXPECT_DOUBLE_EQ(*parse_double("0.5"), 0.5);
    EXPECT_DOUBLE_EQ(*parse_double("1.5M"), 1500000.0);
    EXPECT_DOUBLE_EQ(*parse_double("-2"), -2.0);
    EXPECT_DOUBLE_EQ(*parse_double("1.Kb"), 1024.0);
    for (const char* s : { "", ".", "1..2", "1e5", "inf", "nan", "0x1p3", "1.5X" }) {
        EXPECT_FALSE(parse_double(s).has_value()) << "'" << s << "'";
    }
}

TEST(ParseBool, Spellings) {
    EXPECT_EQ(parse_bool("yes"), true);
    EXPECT_EQ(parse_bool("true"), true);
    EXPECT_EQ(parse_bool("1"), true);
    EXPECT_EQ(parse_bool("no"), false);
    EXPECT_EQ(parse_bool("false"), false);
    EXPECT_EQ(parse_bool("0"), false);
    EXPECT_FALSE(parse_bool("on").has_value());
}

TEST(ParseNumber, TypeRanges) {
    EXPECT_EQ(parse_number<std::size_t>("8Mb"), 8u * 1024 * 1024);
    EXPECT_EQ(parse_number<int>("-1"), -1);
    EXPECT_FALSE(parse_number<int>("3G").has_value());
    EXPECT_FALSE(parse_number<std::uint8_t>("256").has_value());
    EXPECT_FALSE(parse_number<unsigned>("-1").has_value());
    EXPECT_FLOAT_EQ(*parse_number<float>("0.25"), 0.25f);
    EXPECT_EQ(parse_number<bool>("yes"), true);
}
