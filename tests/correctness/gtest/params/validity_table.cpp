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

#include "validity_table.h"

namespace oneccl_test {

bool ValidityTable::is_valid(const CollectiveParams &params,
                             const DeviceCapabilities &device_capabilities,
                             const LaunchTopology &topology, std::string *why) {
    (void)topology; // not consulted by any rule yet

    // DeviceCapabilities refusals
    if (params.dtype == onecclFloat64 && !device_capabilities.supports_fp64) {
        if (why)
            *why = kWhyFp64Unsupported;
        return false;
    }

    if (params.dtype == onecclBfloat16 && !device_capabilities.supports_bf16) {
        if (why)
            *why = kWhyBf16Unsupported;
        return false;
    }

    // AllToAll in-place gives wrong results in the product today
    if (params.collective == Collective::ONECCL_ALLTOALL &&
        params.is_in_place) {
        if (why)
            *why = kWhyAllToAllInPlaceBroken;
        return false;
    }

    // Only AllReduce is implemented so far
    if (params.collective != Collective::ONECCL_ALLREDUCE) {
        if (why)
            *why = kWhyAllReduceOnly;
        return false;
    }

    return true;
}

} // namespace oneccl_test
