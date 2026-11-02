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

#include "rank_constant_pattern.h"

namespace oneccl_test {

// The value does not depend on the element index
void RankConstantPattern::fill_values(int rank, size_t /*begin_index*/,
                                      size_t count, double *out_values) const {
    const double value =
        static_cast<double>(rank) + static_cast<double>(buffer_index_) + 1.0;
    for (size_t index = 0; index < count; ++index) {
        out_values[index] = value;
    }
}

} // namespace oneccl_test
