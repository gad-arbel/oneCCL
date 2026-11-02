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

#include "sequential_order.h"

namespace oneccl_test {

void SequentialOrder::order(int my_rank, size_t contributor_count,
                            std::vector<size_t> &permutation) const {
    (void)my_rank; // unused - same order for all ranks
    permutation.resize(contributor_count);
    for (size_t position = 0; position < contributor_count; ++position) {
        permutation[position] = position;
    }
}

} // namespace oneccl_test
