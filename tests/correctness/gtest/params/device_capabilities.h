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

namespace oneccl_test {

// Device capabilities (what the part can do).
// POD filled by the test environment; this layer never queries the device.
struct DeviceCapabilities {
    bool supports_fp64 = false;
    bool supports_bf16 = false;
    int devices_per_node = 0;
};

// Launch topology (how the launcher laid out the world).
struct LaunchTopology {
    int ranks_per_node = 1;
    int nodes = 1;

    bool has_scaleup() const { return ranks_per_node > 1; }
    bool has_scaleout() const { return nodes > 1; }
};

} // namespace oneccl_test
