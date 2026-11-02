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

#include "bf16.h"
#include "data_spec.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace oneccl_test {

namespace numeric_conversion {

// Convert any supported T to double
template <typename T> double to_double(T value) {
    if constexpr (std::is_same_v<T, Bf16>) {
        return value.to_double();
    } else {
        return static_cast<double>(value);
    }
}

// Convert double to T (single conversion site - matches fill and reference)
template <typename T> T from_double(double value) {
    if constexpr (std::is_same_v<T, Bf16>) {
        return Bf16::from_double(value);
    } else {
        return static_cast<T>(value);
    }
}

} // namespace numeric_conversion

// Type used to hold the running result while folding values of T across a
// reduction. Deduce accumulator type without eagerly instantiating
// make_unsigned for float types
template <typename T, bool = std::is_integral_v<T>> struct FoldAccumulator {
    using type = double;
};
template <typename T> struct FoldAccumulator<T, true> {
    using type = std::make_unsigned_t<T>;
};

// Fold<T> - the single arithmetic home for reduction operations.
// Accumulator: unsigned for integers (wraps by definition), double for fp/bf16.
template <typename T> struct Fold {
    using Accumulator = typename FoldAccumulator<T>::type;

    // Initialize accumulator from first value
    static Accumulator init(T value) {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<Accumulator>(value);
        } else {
            return numeric_conversion::to_double(value);
        }
    }

    // Fold one value into accumulator
    static void step(PatternOp op, Accumulator &accumulator, T value) {
        switch (op) {
        case PatternOp::ONECCL_SUM:
        case PatternOp::ONECCL_PRE_MUL_SUM:
        case PatternOp::ONECCL_AVG:
            if constexpr (std::is_integral_v<T>) {
                // Add in unsigned Accumulator (well-defined wraparound), then
                // narrow to T and widen back to match hardware wrap.
                Accumulator sum = accumulator + static_cast<Accumulator>(value);
                accumulator = static_cast<Accumulator>(static_cast<T>(sum));
            } else {
                accumulator += numeric_conversion::to_double(value);
            }
            break;

        case PatternOp::ONECCL_PROD:
            if constexpr (std::is_integral_v<T>) {
                Accumulator product =
                    accumulator * static_cast<Accumulator>(value);
                accumulator = static_cast<Accumulator>(static_cast<T>(product));
            } else {
                accumulator *= numeric_conversion::to_double(value);
            }
            break;

        case PatternOp::ONECCL_MAX:
            if constexpr (std::is_integral_v<T>) {
                T current = static_cast<T>(accumulator);
                if (value > current)
                    accumulator = static_cast<Accumulator>(value);
            } else {
                accumulator = std::fmax(accumulator,
                                        numeric_conversion::to_double(value));
            }
            break;

        case PatternOp::ONECCL_MIN:
            if constexpr (std::is_integral_v<T>) {
                T current = static_cast<T>(accumulator);
                if (value < current)
                    accumulator = static_cast<Accumulator>(value);
            } else {
                accumulator = std::fmin(accumulator,
                                        numeric_conversion::to_double(value));
            }
            break;
        }
    }

    // Finalize accumulator to result
    static T finish(PatternOp op, Accumulator accumulator,
                    size_t contributor_count) {
        if (op == PatternOp::ONECCL_AVG) {
            if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(accumulator) /
                       static_cast<T>(contributor_count);
            } else {
                return numeric_conversion::from_double<T>(
                    accumulator / static_cast<double>(contributor_count));
            }
        }
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(accumulator);
        } else {
            return numeric_conversion::from_double<T>(accumulator);
        }
    }

    // Accumulate absolute value (for tolerance bound)
    static void accumulate_absolute(double &absolute_sum, T value) {
        absolute_sum += std::fabs(numeric_conversion::to_double(value));
    }
};

} // namespace oneccl_test
