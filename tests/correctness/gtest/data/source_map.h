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

#include <cstddef>
#include <vector>

namespace oneccl_test {

// Who contributes to each element
enum class Contributors {
    ONECCL_ALL_RANKS, // AllReduce, Reduce, ReduceScatter (fold over all ranks)
    ONECCL_RANK_PER_SEGMENT, // AllGather, AllToAll
    ONECCL_SINGLE_RANK       // Broadcast, Scatter (root only, no fold)
};

// Per-collective knowledge for locating source values.
// TestDataImpl::check_invariants() enforces the cross-field shape invariants
// on these fields at runtime.
struct SourceMap {
    Contributors contributors = Contributors::ONECCL_ALL_RANKS;
    size_t segment_element_count = 0; // elements per segment (uniform shapes)
    size_t source_base = 0;           // base offset in send buffer, in elements
    int root = -1;                    // root rank (single-rank contributors)
    std::vector<size_t> segment_offsets; // segment boundaries (non-uniform)
    std::vector<size_t> source_offsets;  // per-rank source base (AllToAllV)
};

} // namespace oneccl_test
