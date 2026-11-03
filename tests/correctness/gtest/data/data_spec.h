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
#include <cstddef>
#include <vector>

namespace oneccl_test {

// Pattern kind for test data generation
enum class PatternKind {
    ONECCL_RANK_CONSTANT, // rank + buffer_index + 1
    ONECCL_TRIANGLE,      // triangle wave
    ONECCL_MOD256         // (rank*count + index) % M(T)
};

// Reduction order for fold
enum class OrderKind {
    ONECCL_SEQUENTIAL // ranks 0..rank_count-1 in order
};

// Pattern operation (NOT onecclRedOp_t - PreMulSum is runtime-allocated)
enum class PatternOp {
    ONECCL_SUM,
    ONECCL_PROD,
    ONECCL_MAX,
    ONECCL_MIN,
    ONECCL_AVG,
    ONECCL_PRE_MUL_SUM
};

// Comparison mode
enum class CompareMode {
    ONECCL_BIT_EXACT, // memcmp or per-element ==
    ONECCL_TOLERANCE  // absolute tolerance bound
};

// Complete specification for one buffer pair's (send/recv) test data
struct DataSpec {
    onecclDataType_t dtype = onecclFloat32;
    PatternOp op = PatternOp::ONECCL_SUM;
    size_t count = 0;              // API argument, per-rank
    size_t send_element_count = 0; // send buffer bound
    size_t recv_element_count = 0; // receive buffer element count
    int my_rank = 0;
    size_t buffer_index = 0; // which of buffer_count pairs this spec describes
    std::vector<int> comm_ranks;
    std::vector<double>
        pre_mul_scalars; // one per position, empty for non-PreMulSum
    PatternKind pattern = PatternKind::ONECCL_RANK_CONSTANT;
    OrderKind order = OrderKind::ONECCL_SEQUENTIAL;
};

} // namespace oneccl_test
