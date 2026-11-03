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

#include "param_expander.h"
#include "../data/test_data.h"
#include "validity_table.h"
#include <algorithm>

namespace oneccl_test {

std::vector<CollectiveParams> ParamExpander::expand(
    Collective collective, const std::vector<onecclDataType_t> &dtypes,
    const std::vector<PatternOp> &ops, const std::vector<size_t> &counts,
    const DeviceCapabilities &device_capabilities,
    const LaunchTopology &topology, std::vector<Refusal> *refusals,
    bool is_in_place) {

    std::vector<CollectiveParams> result;

    // Placeholder communicator sized to the actual launch, not a hardcoded
    // {0, 1} - keeps the overflow check in sync with real world_size.
    const int placeholder_world_size =
        std::max(1, topology.ranks_per_node * topology.nodes);
    std::vector<int> placeholder_ranks(
        static_cast<size_t>(placeholder_world_size));
    for (int rank = 0; rank < placeholder_world_size; ++rank) {
        placeholder_ranks[static_cast<size_t>(rank)] = rank;
    }

    // Cartesian product: dtype × op × count
    for (auto dtype : dtypes) {
        for (auto op : ops) {
            for (auto count : counts) {
                CollectiveParams params(collective, dtype, op, count);
                params.is_in_place = is_in_place;

                // Check validity
                std::string why;
                if (!ValidityTable::is_valid(params, device_capabilities,
                                             topology, &why)) {
                    if (refusals)
                        refusals->push_back({params, why});
                    continue;
                }

                // Check numeric support against the placeholder communicator,
                // for every buffer pair: the pattern grows with buffer_index
                bool is_supported = true;
                for (size_t buffer_index = 0;
                     is_supported && buffer_index < params.buffer_count;
                     ++buffer_index) {
                    DataSpec spec =
                        params.to_data_spec(0, placeholder_ranks, buffer_index);
                    is_supported = ITestData::is_supported(spec, &why);
                }
                if (!is_supported) {
                    if (refusals)
                        refusals->push_back({params, why});
                    continue;
                }

                result.push_back(params);
            }
        }
    }

    return result;
}

} // namespace oneccl_test
