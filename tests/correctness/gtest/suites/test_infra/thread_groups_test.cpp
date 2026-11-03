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

// single_thread_group() and by_parity() are pure functions of the thread count.

#include "framework/thread_groups.h"

#include <gtest/gtest.h>
#include <vector>

using namespace oneccl_test;

TEST(ThreadGroupsTest,
     single_thread_group_with_n_threads_then_one_group_with_every_thread) {
    const ThreadGroups one = single_thread_group(1);
    ASSERT_EQ(1u, one.size());
    EXPECT_EQ(std::vector<int>({0}), one[0].threads);

    const ThreadGroups four = single_thread_group(4);
    ASSERT_EQ(1u, four.size());
    EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), four[0].threads);
    EXPECT_EQ(0u, four[0].index);

    EXPECT_TRUE(single_thread_group(0).empty());
}

TEST(ThreadGroupsTest, by_parity_with_two_threads_then_splits_even_and_odd) {
    const ThreadGroups two = by_parity(2);
    ASSERT_EQ(2u, two.size());
    EXPECT_EQ(std::vector<int>({0}), two[0].threads);
    EXPECT_EQ(std::vector<int>({1}), two[1].threads);
    EXPECT_EQ(0u, two[0].index);
    EXPECT_EQ(1u, two[1].index);

    const ThreadGroups five = by_parity(5);
    ASSERT_EQ(2u, five.size());
    EXPECT_EQ(std::vector<int>({0, 2, 4}), five[0].threads);
    EXPECT_EQ(std::vector<int>({1, 3}), five[1].threads);
}

TEST(ThreadGroupsTest, by_parity_with_one_thread_then_drops_the_empty_group) {
    const ThreadGroups one = by_parity(1);
    ASSERT_EQ(1u, one.size());
    EXPECT_EQ(std::vector<int>({0}), one[0].threads);

    EXPECT_TRUE(by_parity(0).empty());
}
