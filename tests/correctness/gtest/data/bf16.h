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

#include <cstdint>
#include <cstring>

namespace oneccl_test {

// Minimal bfloat16 type for data/ layer — no SYCL dependency.
// BFloat16 is the top 16 bits of a float32 (same exponent, rounded mantissa).
struct Bf16 {
    // Bits dropped/restored when converting to/from float32.
    static constexpr uint32_t kShiftBits = 16;
    // Round-to-nearest-even tie bias: half of the discarded range, minus one.
    static constexpr uint32_t kRoundToNearestEvenBias =
        (1u << (kShiftBits - 1)) - 1u;

    uint16_t bits = 0;

    // Exponent and mantissa masks of a float32 bit pattern, and the bf16
    // quiet-NaN bit.
    static constexpr uint32_t kFloatExponentMask = 0x7F800000u;
    static constexpr uint32_t kFloatMantissaMask = 0x007FFFFFu;
    static constexpr uint16_t kQuietNanBit = 0x0040u;

    // Convert from double/float, rounding to nearest even (matches
    // sycl::ext::oneapi::bfloat16 and the AVX512BF16 hardware path).
    static Bf16 from_float(float value) {
        uint32_t bit_pattern;
        std::memcpy(&bit_pattern, &value, sizeof(bit_pattern));
        // NaN must stay NaN: the rounding bias would carry a low-only payload
        // into +Inf, or an all-ones one into the sign bit
        if ((bit_pattern & kFloatExponentMask) == kFloatExponentMask &&
            (bit_pattern & kFloatMantissaMask) != 0u) {
            Bf16 nan;
            nan.bits = static_cast<uint16_t>(
                static_cast<uint16_t>(bit_pattern >> kShiftBits) |
                kQuietNanBit);
            return nan;
        }
        uint32_t retained_lsb = (bit_pattern >> kShiftBits) & 1u;
        uint32_t rounding_bias = kRoundToNearestEvenBias + retained_lsb;
        Bf16 result;
        result.bits =
            static_cast<uint16_t>((bit_pattern + rounding_bias) >> kShiftBits);
        return result;
    }

    static Bf16 from_double(double value) {
        return from_float(static_cast<float>(value));
    }

    // Convert to float
    float to_float() const {
        uint32_t bit_pattern = static_cast<uint32_t>(bits) << kShiftBits;
        float result;
        std::memcpy(&result, &bit_pattern, sizeof(result));
        return result;
    }

    double to_double() const { return static_cast<double>(to_float()); }

    // Comparison needed for Fold min/max
    bool operator<(Bf16 other) const { return to_float() < other.to_float(); }
    bool operator>(Bf16 other) const { return to_float() > other.to_float(); }
    bool operator==(Bf16 other) const { return bits == other.bits; }
    bool operator!=(Bf16 other) const { return bits != other.bits; }
};

} // namespace oneccl_test
