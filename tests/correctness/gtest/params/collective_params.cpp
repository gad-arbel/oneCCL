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

#include "collective_params.h"

namespace oneccl_test {

DataSpec CollectiveParams::to_data_spec(int my_rank,
                                        const std::vector<int> &comm_ranks,
                                        size_t buffer_index) const {
    DataSpec spec;
    spec.dtype = dtype;
    spec.op = op;
    spec.count = count;
    spec.my_rank = my_rank;
    spec.buffer_index = buffer_index;
    spec.comm_ranks = comm_ranks;
    spec.pattern = pattern;
    spec.order = order;

    // AllReduce sizing: send_element_count and recv_element_count both = count
    // (the only collective supported so far)
    spec.send_element_count = count;
    spec.recv_element_count = count;

    // PreMulSum scalars (derived from position)
    if (op == PatternOp::ONECCL_PRE_MUL_SUM) {
        spec.pre_mul_scalars.resize(comm_ranks.size());
        for (size_t position = 0; position < comm_ranks.size(); ++position) {
            // Simple formula: scalar = rank + 1
            spec.pre_mul_scalars[position] =
                static_cast<double>(comm_ranks[position] + 1);
        }
    }

    return spec;
}

} // namespace oneccl_test
