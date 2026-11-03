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

#include "data_pattern.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace oneccl_test {

double IDataPattern::max_absolute_value(const DataSpec &spec) {
    // RankConstant: largest rank + buffer_index + 1. Mod256: M(T) - 1.
    // Triangle: dtype-dependent, a placeholder until the pattern exists.

    // comm_ranks need not be 0..n-1, so take the largest actual rank
    int max_rank = -1;
    for (int rank : spec.comm_ranks) {
        max_rank = std::max(max_rank, rank);
    }

    // A premultiplier widens every contribution, so the largest scalar
    // enters the bound and the overflow refusal covers it with no second rule.
    double scale = 1.0;
    if (spec.op == PatternOp::ONECCL_PRE_MUL_SUM) {
        for (double scalar : spec.pre_mul_scalars) {
            scale = std::max(scale, std::fabs(scalar));
        }
    }

    switch (spec.pattern) {
    case PatternKind::ONECCL_RANK_CONSTANT:
        return scale * (static_cast<double>(max_rank) +
                        static_cast<double>(spec.buffer_index) + 1.0);

    case PatternKind::ONECCL_MOD256: {
        // M(T) = min(256, largest_exact_int + 1)
        // For int8: 128, others: 256
        if (spec.dtype == onecclInt8) {
            return 127.0;
        }
        return 255.0;
    }

    case PatternKind::ONECCL_TRIANGLE:
        // Simplified placeholder until the pattern is implemented
        return 1.0; // fp range

    default:
        return 1.0;
    }
}

} // namespace oneccl_test
