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

namespace oneccl_test {

// The `why` text of each refusal, kept in one place so tests do not repeat it.
inline constexpr const char *kWhyFp64Unsupported = "fp64 unsupported";
inline constexpr const char *kWhyBf16Unsupported = "bf16 unsupported";
inline constexpr const char *kWhyAllToAllInPlaceBroken =
    "AllToAll in-place broken";
inline constexpr const char *kWhyAllReduceOnly =
    "unsupported collective: AllReduce only";

// Structural and hardware-based refusals
class ValidityTable {
  public:
    // True if params are valid for this hardware/topology
    static bool is_valid(const CollectiveParams &params,
                         const DeviceCapabilities &device_capabilities,
                         const LaunchTopology &topology, std::string *why);
};

} // namespace oneccl_test
