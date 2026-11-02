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

#pragma once

#include "oneapi/ccl.h"
#include <utility>
#include <vector>

namespace oneccl_test {

// One world of test threads. id is minted per group per run_threads() call,
// never the process-wide one.
struct ThreadGroup {
    size_t index = 0;
    std::vector<int> threads;
    onecclUniqueId id{};
};

using ThreadGroups = std::vector<ThreadGroup>;

// Every thread in one group. The default. No thread gives no group.
inline ThreadGroups single_thread_group(int n) {
    if (n <= 0) {
        return {};
    }
    ThreadGroup group;
    for (int thread = 0; thread < n; ++thread) {
        group.threads.push_back(thread);
    }
    return {group};
}

// Even and odd thread indices in two groups, so {{0, 2, 4}, {1, 3}} at n = 5.
// A group with no thread is dropped, so n = 1 gives {{0}}.
inline ThreadGroups by_parity(int n) {
    ThreadGroups groups(2);
    for (int thread = 0; thread < n; ++thread) {
        groups[static_cast<size_t>(thread % 2)].threads.push_back(thread);
    }
    ThreadGroups populated_groups;
    for (ThreadGroup &group : groups) {
        if (!group.threads.empty()) {
            group.index = populated_groups.size();
            populated_groups.push_back(std::move(group));
        }
    }
    return populated_groups;
}

} // namespace oneccl_test
