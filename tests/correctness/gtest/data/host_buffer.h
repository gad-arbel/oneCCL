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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace oneccl_test {

// Host-side byte buffer for test data.
// Compiles into occl_gtest_data with no SYCL/MPI dependencies. Device results
// reach it only through the device-copy layer, which owns the sync.
class HostBuffer {
  public:
    // Allocate a zero-filled buffer of given size
    explicit HostBuffer(size_t bytes);

    // Take ownership of host vector
    explicit HostBuffer(std::vector<std::byte> host_values);

    // Raw byte access
    std::byte *data();
    const std::byte *data() const;

    // Size in bytes
    size_t size() const;

    // Typed view (no bounds check - caller ensures alignment/size). The
    // storage comes from operator new, so it is aligned for every ccl dtype.
    template <typename T> const T *as() const {
        return reinterpret_cast<const T *>(data());
    }

    // Non-const overload, so a writer doesn't need a const_cast (UB if the
    // caller ever holds a genuinely const HostBuffer).
    template <typename T> T *as() { return reinterpret_cast<T *>(data()); }

  private:
    std::vector<std::byte> storage_;
};

} // namespace oneccl_test
