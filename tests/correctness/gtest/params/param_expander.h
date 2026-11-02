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

#include "collective_params.h"
#include "device_capabilities.h"
#include <string>
#include <vector>

namespace oneccl_test {

// Parameter matrix expansion
class ParamExpander {
  public:
    // One refused row, with the reason it was refused.
    struct Refusal {
        CollectiveParams params;
        std::string why;
    };

    // Expand one collective's parameter matrix.
    //
    // `refusals`, if non-null, has refused rows appended to it (never
    // cleared) - passing the SAME vector across multiple expand() calls (one
    // per collective) accumulates refusals for the whole matrix, since each
    // call only appends.
    //
    // `is_in_place` marks every row in place before it is judged, so the
    // validity rules see the row that will run.
    static std::vector<CollectiveParams>
    expand(Collective collective, const std::vector<onecclDataType_t> &dtypes,
           const std::vector<PatternOp> &ops, const std::vector<size_t> &counts,
           const DeviceCapabilities &device_capabilities,
           const LaunchTopology &topology,
           std::vector<Refusal> *refusals = nullptr, bool is_in_place = false);
};

} // namespace oneccl_test
