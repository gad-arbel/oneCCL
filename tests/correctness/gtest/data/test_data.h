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

#include "compare/mismatch_report.h"
#include "data_spec.h"
#include "host_buffer.h"
#include "source_map.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace oneccl_test {

// Abstract base for test data generation and verification.
// Type-erases dtype via create() factory (single switch in the tree).
// One instance describes one rank's buffer pair. compare() and verify() keep
// per-instance scratch and the next expected offset, so an instance must not
// be shared by concurrent callers.
class ITestData {
  public:
    // Caps how many elements are materialized into scratch buffers at once,
    // so compare() doesn't need the full expected buffer in memory for large
    // collectives.
    static constexpr size_t kWindowElementCount = 64 * 1024;
    static constexpr std::byte kPoisonByte{0xA5};

    virtual ~ITestData() = default;

    // Factory: create TestDataImpl<T> based on dtype. Returns nullptr for a
    // pattern, order or dtype not implemented yet, and throws
    // std::logic_error when my_rank is not in comm_ranks.
    static std::unique_ptr<ITestData> create(const DataSpec &spec);

    // Is this spec supported? Refuses unimplemented patterns, orders and
    // dtypes, integer avg, overflowing rows, prod/avg tolerance rows and rows
    // whose tolerance window would be too wide.
    static bool is_supported(const DataSpec &spec, std::string *why);

    // Comparison mode for this spec (bit-exact vs tolerance)
    static CompareMode compare_mode(const DataSpec &spec);

    // Unit roundoff u (half an ULP) of dtype; 0 for integer dtypes.
    static double unit_roundoff(onecclDataType_t dtype);

    // ONECCL_TOLERANCE window for one element:
    // gamma(n-1) * absolute_sum + u * |expected|, gamma(k) = k*u / (1 - k*u).
    // Only meaningful for rows is_supported() accepts, i.e. (n-1)*u < 1/2.
    static double tolerance_bound(onecclDataType_t dtype, size_t rank_count,
                                  double absolute_sum, double expected_value);

    // Fill send buffer with this rank's pattern values
    virtual void fill_send(HostBuffer &buffer) const = 0;

    // Poison recv buffer with kPoisonByte (unconditional byte fill)
    virtual void poison_recv(HostBuffer &buffer) const = 0;

    // Assert recv buffer is still poisoned (collective declined/no-op)
    virtual void assert_still_poisoned(const HostBuffer &buffer) const = 0;

    // Compare buffer against expected, return mismatch count (gtest-free).
    // Checks the cross-field invariants on source_map (only at offset == 0);
    // see TestDataImpl::check_invariants(). `offset` is in elements, not
    // bytes.
    virtual size_t compare(const HostBuffer &buffer,
                           const SourceMap &source_map, size_t offset,
                           MismatchReport &report) const = 0;

    // Compare buffer and record a gtest failure with the capped report on
    // mismatch (gtest-facing wrapper)
    virtual void verify(const HostBuffer &buffer, const SourceMap &source_map,
                        size_t offset) const = 0;
};

} // namespace oneccl_test
