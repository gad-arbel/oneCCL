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

#include "../data/data_spec.h"
#include "oneapi/ccl.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace oneccl_test {

// The one list of collectives: the enum and kAllCollectives both expand it, so
// a new collective cannot reach one and miss the other.
#define ONECCL_COLLECTIVES(X)                                                  \
    X(ONECCL_ALLREDUCE)                                                        \
    X(ONECCL_REDUCE)                                                           \
    X(ONECCL_BROADCAST)                                                        \
    X(ONECCL_ALLGATHER)                                                        \
    X(ONECCL_ALLTOALL)                                                         \
    X(ONECCL_ALLTOALLV)                                                        \
    X(ONECCL_REDUCE_SCATTER)                                                   \
    X(ONECCL_SCATTER)                                                          \
    X(ONECCL_SEND_RECV)

// Collective operations
enum class Collective {
#define ONECCL_COLLECTIVE_ENUMERATOR(name) name,
    ONECCL_COLLECTIVES(ONECCL_COLLECTIVE_ENUMERATOR)
#undef ONECCL_COLLECTIVE_ENUMERATOR
};

// Every Collective enumerator, in declaration order, for tests that walk the
// enum.
inline constexpr Collective kAllCollectives[] = {
#define ONECCL_COLLECTIVE_ENTRY(name) Collective::name,
    ONECCL_COLLECTIVES(ONECCL_COLLECTIVE_ENTRY)
#undef ONECCL_COLLECTIVE_ENTRY
};

// The enumerator without its ONECCL_ prefix, lower case: "allreduce".
inline std::string collective_name(Collective collective) {
    std::string name;
    switch (collective) {
#define ONECCL_COLLECTIVE_NAME(enumerator)                                     \
    case Collective::enumerator:                                               \
        name = #enumerator;                                                    \
        break;
        ONECCL_COLLECTIVES(ONECCL_COLLECTIVE_NAME)
#undef ONECCL_COLLECTIVE_NAME
    }
    name.erase(0, std::string("ONECCL_").size());
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
}

inline std::string dtype_name(onecclDataType_t dtype) {
    switch (dtype) {
    case onecclFloat32:
        return "fp32";
    case onecclBfloat16:
        return "bf16";
    case onecclInt32:
        return "int32";
    default:
        return "dt" + std::to_string(static_cast<int>(dtype));
    }
}

inline std::string op_name(PatternOp op) {
    switch (op) {
    case PatternOp::ONECCL_SUM:
        return "sum";
    case PatternOp::ONECCL_PROD:
        return "prod";
    case PatternOp::ONECCL_MAX:
        return "max";
    case PatternOp::ONECCL_MIN:
        return "min";
    case PatternOp::ONECCL_AVG:
        return "avg";
    case PatternOp::ONECCL_PRE_MUL_SUM:
        return "premulsum";
    }
    return "op" + std::to_string(static_cast<int>(op));
}

// AllToAllV count rule
enum class CountRule {
    ONECCL_UNIFORM,        // count for all pairs
    ONECCL_RANK_SCALED,    // source + destination
    ONECCL_RANK_DESCENDING // count - source (clamped)
};

// Complete parameter set for one collective operation.
// A named struct, not a tuple.
struct CollectiveParams {
    CollectiveParams(Collective collective, onecclDataType_t dtype,
                     PatternOp op, size_t count)
        : collective(collective), dtype(dtype), op(op), count(count) {}

    Collective collective;
    onecclDataType_t dtype;
    PatternOp op;
    size_t count; // API argument, per-rank
    CountRule count_rule = CountRule::ONECCL_UNIFORM;
    size_t buffer_count = 1; // {1, 4} independent buffer pairs
    size_t iterations = 1;   // execute() loop count
    int root = 0;            // for root-based collectives
    int peer = -1;           // for SendRecv
    bool is_in_place = false;
    PatternKind pattern = PatternKind::ONECCL_RANK_CONSTANT;
    OrderKind order = OrderKind::ONECCL_SEQUENTIAL;

    // Convert to DataSpec for one buffer pair
    DataSpec to_data_spec(int my_rank, const std::vector<int> &comm_ranks,
                          size_t buffer_index) const;
};

} // namespace oneccl_test
