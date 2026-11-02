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

#include "test_data.h"
#include "bf16.h"
#include "orders/sequential_order.h"
#include "patterns/data_pattern.h"
#include "patterns/rank_constant_pattern.h"
#include "test_data_impl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace oneccl_test {

// The single dtype *type-instantiation* switch in the tree: resolves DataSpec
// into a concrete TestDataImpl<T> (defined in test_data_impl.h), plus the
// non-template validity/compare-mode logic that doesn't need to be templated
// on T.

// Per-dtype maximum absolute value representable in T, used by
// is_supported()'s overflow refusals.
static constexpr double kInt8MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<int8_t>::max());
static constexpr double kUint8MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<uint8_t>::max());
static constexpr double kInt32MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<int32_t>::max());
static constexpr double kUint32MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<uint32_t>::max());
static constexpr double kInt64MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<int64_t>::max());
static constexpr double kUint64MaxAbsoluteValue =
    static_cast<double>(std::numeric_limits<uint64_t>::max());
static constexpr double kFloat32MaxAbsoluteValue =
    std::numeric_limits<float>::max();
static constexpr double kFloat64MaxAbsoluteValue =
    std::numeric_limits<double>::max();
// IEEE-754 half-precision max. No C++ fp16 type exists in this codebase, so
// unlike the others above this one stays a documented literal instead of a
// std::numeric_limits<T>::max().
static constexpr double kFloat16MaxAbsoluteValue = 65504.0;
// bf16 shares fp32's exponent range, so it reuses fp32's bound.
static constexpr double kBfloat16MaxAbsoluteValue = kFloat32MaxAbsoluteValue;

// Returns the maximum absolute value representable in the C++ type `dtype`
// maps to. Covers all ten onecclDataType_t values - distinct from the dtype
// *type-instantiation* switch above (ITestData::create()'s), which only needs
// to dispatch the dtypes create() currently instantiates; this is a plain
// value lookup with no type parameter.
static double max_absolute_value_for_dtype(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclInt8:
        return kInt8MaxAbsoluteValue;
    case onecclUint8:
        return kUint8MaxAbsoluteValue;
    case onecclInt32:
        return kInt32MaxAbsoluteValue;
    case onecclUint32:
        return kUint32MaxAbsoluteValue;
    case onecclInt64:
        return kInt64MaxAbsoluteValue;
    case onecclUint64:
        return kUint64MaxAbsoluteValue;
    case onecclFloat16:
        return kFloat16MaxAbsoluteValue;
    case onecclFloat32:
        return kFloat32MaxAbsoluteValue;
    case onecclFloat64:
        return kFloat64MaxAbsoluteValue;
    case onecclBfloat16:
        return kBfloat16MaxAbsoluteValue;
    // Unreachable: all ten dtype values are handled above. A default is
    // still needed because onecclChar/onecclInt/onecclHalf/onecclFloat/
    // onecclDouble are separate enumerator names aliasing the values above.
    default:
        return kFloat64MaxAbsoluteValue;
    }
}

// True for the dtypes ITestData::create() instantiates; keep in sync with its
// switch (TestDataTest.IsSupportedMatchesCreate pins the two together)
static bool is_instantiated_dtype(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclFloat32:
    case onecclBfloat16:
    case onecclInt32:
        return true;
    default:
        return false;
    }
}

// True for the six integer dtypes, which fold with wraparound and compare
// bit-exact
static bool is_integer_dtype(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclInt8:
    case onecclUint8:
    case onecclInt32:
    case onecclUint32:
    case onecclInt64:
    case onecclUint64:
        return true;
    default:
        return false;
    }
}

// A tolerance row is refused once gamma(n-1) >= 1, i.e. (n-1)*u >= 1/2: the
// window then covers the whole result, so a zeroed output would pass.
static constexpr double kMaxToleranceRoundingProduct = 0.5;

// Helper to create pattern based on spec
static std::unique_ptr<IDataPattern> create_pattern(const DataSpec &spec) {
    switch (spec.pattern) {
    case PatternKind::ONECCL_RANK_CONSTANT:
        return std::make_unique<RankConstantPattern>(spec.buffer_index);
    // Only RankConstant is implemented so far
    default:
        return nullptr;
    }
}

// Helper to create order based on spec
static std::unique_ptr<IReductionOrder> create_order(const DataSpec &spec) {
    switch (spec.order) {
    case OrderKind::ONECCL_SEQUENTIAL:
        return std::make_unique<SequentialOrder>();
    // Only Sequential is implemented so far
    default:
        return nullptr;
    }
}

std::unique_ptr<ITestData> ITestData::create(const DataSpec &spec) {
    // Create pattern and order
    auto pattern = create_pattern(spec);
    auto order = create_order(spec);
    if (!pattern || !order) {
        return nullptr;
    }

    // Get comparison mode
    CompareMode mode = compare_mode(spec);

    // The single dtype *type-instantiation* switch in the tree
    switch (spec.dtype) {
    case onecclFloat32:
        return std::make_unique<TestDataImpl<float>>(spec, std::move(pattern),
                                                     std::move(order), mode);
    case onecclBfloat16:
        return std::make_unique<TestDataImpl<Bf16>>(spec, std::move(pattern),
                                                    std::move(order), mode);
    case onecclInt32:
        return std::make_unique<TestDataImpl<int32_t>>(spec, std::move(pattern),
                                                       std::move(order), mode);
    // Only fp32, bf16 and int32 are instantiated so far
    default:
        return nullptr;
    }
}

bool ITestData::is_supported(const DataSpec &spec, std::string *why) {
    size_t rank_count = spec.comm_ranks.size();
    double max_value = IDataPattern::max_absolute_value(spec);

    // Only what create() actually wires up is supported
    if (spec.pattern != PatternKind::ONECCL_RANK_CONSTANT) {
        if (why)
            *why = "pattern not implemented";
        return false;
    }
    if (spec.order != OrderKind::ONECCL_SEQUENTIAL) {
        if (why)
            *why = "order not implemented";
        return false;
    }
    if (!is_instantiated_dtype(spec.dtype)) {
        if (why)
            *why = "datatype not implemented";
        return false;
    }

    // Integer avg has undefined rounding
    if (spec.op == PatternOp::ONECCL_AVG && is_integer_dtype(spec.dtype)) {
        if (why)
            *why = "avg on integer dtype (rounding undefined)";
        return false;
    }

    // Overflow checks
    if (spec.op == PatternOp::ONECCL_SUM ||
        spec.op == PatternOp::ONECCL_PRE_MUL_SUM ||
        spec.op == PatternOp::ONECCL_AVG) {
        // rank_count * max_absolute_value must fit in T
        double sum_bound = static_cast<double>(rank_count) * max_value;
        double type_max_value = max_absolute_value_for_dtype(spec.dtype);

        if (sum_bound > type_max_value) {
            if (why)
                *why = "sum overflow";
            return false;
        }
    }

    if (spec.op == PatternOp::ONECCL_PROD) {
        // max_absolute_value^rank_count must fit in T
        double product_bound =
            std::pow(max_value, static_cast<double>(rank_count));
        double type_max_value = max_absolute_value_for_dtype(spec.dtype);

        if (product_bound > type_max_value) {
            if (why)
                *why = "prod overflow";
            return false;
        }
    }

    // tolerance_bound() has the sum-shaped window only; prod and avg need
    // their own bounds, so refuse their tolerance rows until those exist.
    if ((spec.op == PatternOp::ONECCL_PROD ||
         spec.op == PatternOp::ONECCL_AVG) &&
        compare_mode(spec) == CompareMode::ONECCL_TOLERANCE) {
        if (why)
            *why = "tolerance bound not implemented for prod/avg";
        return false;
    }

    // A tolerance window with gamma(n-1) >= 1 covers the whole result, so a
    // zeroed output would pass: refuse the row rather than compare it.
    if (rank_count > 1 && compare_mode(spec) == CompareMode::ONECCL_TOLERANCE &&
        static_cast<double>(rank_count - 1) * unit_roundoff(spec.dtype) >=
            kMaxToleranceRoundingProduct) {
        if (why)
            *why = "tolerance window too wide at this rank count";
        return false;
    }

    return true;
}

// std::ldexp(x, exp) = x * 2^exp
double ITestData::unit_roundoff(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclFloat16:
        return std::ldexp(1.0, -11);
    case onecclBfloat16:
        return std::ldexp(1.0, -8);
    case onecclFloat32:
        return std::ldexp(1.0, -24);
    case onecclFloat64:
        return std::ldexp(1.0, -53);
    default:
        return 0.0; // integers are bit-exact
    }
}

double ITestData::tolerance_bound(onecclDataType_t dtype, size_t rank_count,
                                  double absolute_sum, double expected_value) {
    const double u = unit_roundoff(dtype);
    const double dependent_roundings =
        rank_count > 1 ? static_cast<double>(rank_count - 1) : 0.0;
    const double growth =
        dependent_roundings * u / (1.0 - dependent_roundings * u);
    // The second term charges the reference's own double -> T cast.
    return growth * absolute_sum + u * std::fabs(expected_value);
}

CompareMode ITestData::compare_mode(const DataSpec &spec) {
    // Min/max always bit-exact: nothing rounds
    if (spec.op == PatternOp::ONECCL_MIN || spec.op == PatternOp::ONECCL_MAX) {
        return CompareMode::ONECCL_BIT_EXACT;
    }

    // Integer ops always bit-exact (Fold wraps identically)
    if (is_integer_dtype(spec.dtype)) {
        return CompareMode::ONECCL_BIT_EXACT;
    }

    // The closed forms below are RankConstant's; other patterns compare with
    // tolerance until they bring their own.
    if (spec.pattern != PatternKind::ONECCL_RANK_CONSTANT) {
        return CompareMode::ONECCL_TOLERANCE;
    }

    // Integers up to 1/u are exact in a binary float: 2^8 for bf16, 2^11 for
    // fp16, 2^24 for fp32, 2^53 for fp64.
    const double exact_integer_range = 1.0 / unit_roundoff(spec.dtype);
    const size_t positions = spec.comm_ranks.size();

    // Every partial result of a sum or product of positive integers is at most
    // the final one, so the row is bit-exact in any fold order exactly when
    // every contribution is an integer and the final result is in range.
    // RankConstant's per-rank value is rank + buffer_index + 1, taken from the
    // actual comm_ranks, which need not be 0..n-1.
    if (spec.op == PatternOp::ONECCL_SUM ||
        spec.op == PatternOp::ONECCL_PRE_MUL_SUM) {
        const bool is_pre_mul_sum = spec.op == PatternOp::ONECCL_PRE_MUL_SUM;
        double exact_sum = 0.0;
        for (size_t position = 0; position < positions; ++position) {
            double scalar = 1.0;
            if (is_pre_mul_sum) {
                if (position >= spec.pre_mul_scalars.size()) {
                    return CompareMode::ONECCL_TOLERANCE;
                }
                scalar = spec.pre_mul_scalars[position];
            }
            // A fractional scalar makes the scaled contribution round in T
            if (scalar != std::trunc(scalar)) {
                return CompareMode::ONECCL_TOLERANCE;
            }
            exact_sum += std::fabs(scalar) *
                         (static_cast<double>(spec.comm_ranks[position]) +
                          static_cast<double>(spec.buffer_index) + 1.0);
        }
        if (exact_sum <= exact_integer_range) {
            return CompareMode::ONECCL_BIT_EXACT;
        }
    }

    if (spec.op == PatternOp::ONECCL_PROD) {
        double exact_product = 1.0;
        for (size_t position = 0; position < positions; ++position) {
            exact_product *= static_cast<double>(spec.comm_ranks[position]) +
                             static_cast<double>(spec.buffer_index) + 1.0;
        }
        if (exact_product <= exact_integer_range) {
            return CompareMode::ONECCL_BIT_EXACT;
        }
    }

    // Default: tolerance (avg divides, so it is never assumed exact)
    return CompareMode::ONECCL_TOLERANCE;
}

} // namespace oneccl_test
