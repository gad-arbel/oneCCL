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

// Registry of the selection policy's leaf knobs.
//
// - Leaf knob: a `"<name>=<value>": {}` child of an `algorithm=` node, handed
//   back in LeafDecision::params. It is either value-only (the caller reads it,
//   e.g. ll_table) or bound to a knob_env field (env_binding), which is set
//   per call through restore_policy_knobs() and apply_policy_knobs().
//
// The engine checks every knob of a loaded policy against this registry and
// warns about unknown or invalid ones; they are still loaded. A bound knob
// wins over its env var, even one set in the environment.
//

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/env/env.hpp"
#include "common/utils/parse_utils.hpp"

namespace ccl {
namespace selection {

struct LeafDecision;

enum class knob_type { string, boolean, integer, size, real, choice };

// Bit mask; `any` until a phase= branch is seen.
enum class knob_phase : std::uint8_t { scaleup = 1, scaleout = 2, any = 3 };

// Ties a leaf knob to one knob_env field and the env var that field relates to.
struct env_binding {
    // e.g. "CCL_SYCL_COPY_ENGINE". Usually the variable that sets the bound field. For a
    // knob whose field no variable sets — simple_chunk_size, written only by a policy — it
    // names the spec the field stands in for instead, a different field of a different
    // type. Only the applied-knobs log text reads this.
    const char* cvar = nullptr;
    std::string (*get)(const knob_env&) = nullptr; // current value as text
    bool (*set)(knob_env&, const std::string&) = nullptr; // false if the text doesn't parse

    explicit operator bool() const {
        return cvar && get && set;
    }
};

struct leaf_knob_spec {
    const char* name;
    knob_type type;
    knob_phase phase;
    std::vector<std::string> collectives; // policy root names ("allgather", ...); empty: all
    std::vector<std::string> choices; // knob_type::choice
    env_binding env; // empty: value-only, unless env_per_collective is set
    // Per policy root name, used in place of `env` for that collective.
    std::unordered_map<std::string, env_binding> env_per_collective = {};
};

// The registry, defined at the end of this file.
inline const std::vector<leaf_knob_spec>& leaf_knobs();
const leaf_knob_spec* find_leaf_knob(std::string_view name);

// Whether `value` parses as `type` (and is one of `choices` for a choice).
bool is_valid_knob_value(knob_type type,
                         const std::vector<std::string>& choices,
                         const std::string& value);

// True if the env var is present in the environment, whatever its value.
bool is_env_set(const char* cvar);

// Per call, for env-bound leaf knobs of `phase` (the knob writes the field in
// this thread's call_env() for the duration of the call). restore_policy_knobs()
// puts every bound field back to its startup value; call it before
// apply_policy_knobs() so a knob of an earlier call doesn't stay set. It also
// settles whether the knobs apply, for its phase and until the next lookup:
// from the first lookup after a multi-thread communicator appears, it does
// nothing and call_env() is the global env_data's knobs again.
void restore_policy_knobs(knob_phase phase);
// Applies the knobs of `decision` that apply to `collective`, none when
// restore_policy_knobs() turned them off for this phase. Returns the applied
// knobs as " name=value (CCL_X) ...", empty if none.
std::string apply_policy_knobs(const LeafDecision& decision,
                               knob_phase phase,
                               const std::string& collective);

// -----------------------------------------------------------------------------
// Binding helper: bind_env<&knob_env::field>(CCL_X) for numbers, bools and
// strings.
// -----------------------------------------------------------------------------
namespace knob_detail {

template <class M>
struct member_type;
template <class C, class T>
struct member_type<T C::*> {
    using type = T;
};

template <class T>
std::string to_text(const T& v) {
    if constexpr (std::is_same<T, bool>::value) {
        return v ? "1" : "0";
    }
    else if constexpr (std::is_floating_point<T>::value) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", static_cast<double>(v));
        return buf;
    }
    else if constexpr (std::is_arithmetic<T>::value) {
        return std::to_string(v);
    }
    else {
        return v;
    }
}

inline std::string lower(std::string s) {
    for (auto& c : s) {
        c = utils::parse_detail::to_lower(c);
    }
    return s;
}

} // namespace knob_detail

template <auto Field>
env_binding bind_env(const char* cvar) {
    using T = typename knob_detail::member_type<decltype(Field)>::type;
    env_binding b;
    b.cvar = cvar;
    b.get = [](const knob_env& e) {
        return knob_detail::to_text(e.*Field);
    };
    b.set = [](knob_env& e, const std::string& v) {
        if constexpr (std::is_same<T, std::string>::value) {
            e.*Field = v;
            return true;
        }
        else {
            const auto parsed = utils::parse_number<T>(v);
            if (!parsed) {
                return false;
            }
            e.*Field = *parsed;
            return true;
        }
    };
    return b;
}

// copy_path=copy_engine|kernel|default on a copy-engine switch, where `default` leaves the
// field's startup value. A bool field takes only the first two; an int field is a
// copy_engine_mode, so it also has copy_engine_auto, which get() reports as "auto" and
// set() takes back, letting restore_policy_knobs() return the field to auto.
template <auto Field>
env_binding bind_copy_path(const char* cvar) {
    using T = typename knob_detail::member_type<decltype(Field)>::type;
    env_binding b;
    b.cvar = cvar;
    b.get = [](const knob_env& e) {
        if constexpr (!std::is_same<T, bool>::value) {
            if (e.*Field == copy_engine_auto) {
                return std::string("auto");
            }
        }
        return std::string(e.*Field ? "copy_engine" : "kernel");
    };
    b.set = [](knob_env& e, const std::string& v) {
        const std::string s = knob_detail::lower(v);
        if (s == "copy_engine") {
            e.*Field = static_cast<T>(copy_engine_on);
        }
        else if (s == "kernel") {
            e.*Field = static_cast<T>(copy_engine_kernel);
        }
        else if (s == "auto" && !std::is_same<T, bool>::value) {
            e.*Field = static_cast<T>(copy_engine_auto);
        }
        else if (s != "default") {
            return false;
        }
        return true;
    };
    return b;
}

// -----------------------------------------------------------------------------
// The registry. To add a knob, append a leaf_knob_spec:
//
//   { "<name>",                 // policy key: "<name>=<value>": {} under algorithm=
//     knob_type::<type>,        // value check at load: string, boolean, integer,
//                               // size, real or choice
//     knob_phase::<phase>,      // scaleup or scaleout: the resolver that reads it
//     { "<collective>", ... },  // policy root names ("allgather", ...); {} for all
//     { "<choice>", ... },      // lower-case values for knob_type::choice; else {}
//     <binding>,                // {} or bind_env<...>(...), see below
//     { { "<collective>", <binding> }, ... } },  // optional, see below
//
// and list it under notes.scaleup_knobs / notes.scaleout_knobs in
// policies/collectives_policy.json.
//
// Value-only knob (binding {}): nothing applies it. The resolver reads it from
// the chosen leaf, e.g. decision->knob_size("chunk_bytes") in
// sycl_policy_resolver_json.cpp.
//
// Env-bound knob (bind_env): sets the knob_env field in this thread's
// call_env() for the call, so concurrent calls on other threads keep their own
// values. Declare the field in knob_env (common/env/env.hpp) and parse it into
// env_data::knobs; read it only through call_env() (policy_env.hpp). These
// knobs are off once a multi-thread communicator exists.
//
// Per-collective binding (env_per_collective): for a knob that sets a
// different field per collective. A collective listed there uses its own
// binding; the others use `env`, or nothing when `env` is {}. List these
// collectives in `collectives` too, unless that is {}.
// -----------------------------------------------------------------------------

inline const std::vector<leaf_knob_spec>& leaf_knobs() {
    static const std::vector<leaf_knob_spec> knobs = {
        // LL algorithm per chunk, same format as CCL_SYCL_<COLL>_LL.
        { "ll_table",
          knob_type::string,
          knob_phase::scaleup,
          { "allgather", "allreduce", "reduce_scatter" },
          {},
          {} },
        // Pipeline chunk size of the ring / rabenseifner scaleout algorithms.
        { "chunk_bytes",
          knob_type::size,
          knob_phase::scaleout,
          { "allgather", "allreduce", "reduce_scatter" },
          {},
          {} },
        // Own-block bytes from which the CRI copy-engine self copy runs on a
        // second copy engine (0 = never).
        { "self_copy_threshold",
          knob_type::size,
          knob_phase::scaleup,
          { "allgather", "alltoall", "alltoallv" },
          {},
          {},
          { { "allgather",
              bind_env<&knob_env::sycl_allgatherv_self_copy_threshold>(
                  CCL_SYCL_ALLGATHERV_SELF_COPY_THRESHOLD) },
            { "alltoall",
              bind_env<&knob_env::sycl_alltoall_self_copy_threshold>(
                  CCL_SYCL_ALLTOALL_SELF_COPY_THRESHOLD) },
            { "alltoallv",
              bind_env<&knob_env::sycl_alltoallv_self_copy_threshold>(
                  CCL_SYCL_ALLTOALLV_SELF_COPY_THRESHOLD) } } },
        // Copy engine or compute kernel for the peer copies of a large call.
        { "copy_path",
          knob_type::choice,
          knob_phase::scaleup,
          { "allreduce", "reduce_scatter", "allgather", "alltoall", "alltoallv" },
          { "copy_engine", "kernel", "default" },
          bind_copy_path<&knob_env::sycl_copy_engine>(CCL_SYCL_COPY_ENGINE),
          { { "alltoall",
              bind_copy_path<&knob_env::sycl_alltoall_copy_engine_mode>(
                  CCL_SYCL_ALLTOALL_COPY_ENGINE) },
            { "alltoallv",
              bind_copy_path<&knob_env::sycl_alltoall_copy_engine_mode>(
                  CCL_SYCL_ALLTOALL_COPY_ENGINE) } } },
        // Bytes per pipeline slot of the large ring kernels (tmp-buffer path).
        { "simple_ring_segment_size",
          knob_type::size,
          knob_phase::scaleup,
          { "allreduce", "reduce_scatter", "allgather" },
          {},
          {},
          { { "allreduce",
              bind_env<&knob_env::sycl_allreduce_simple_ring_segment_size>(
                  CCL_SYCL_ALLREDUCE_SIMPLE_RING_SEGMENT_SIZE) },
            { "reduce_scatter",
              bind_env<&knob_env::sycl_reduce_scatter_simple_ring_segment_size>(
                  CCL_SYCL_REDUCE_SCATTER_SIMPLE_RING_SEGMENT_SIZE) },
            { "allgather",
              bind_env<&knob_env::sycl_allgatherv_simple_ring_segment_size>(
                  CCL_SYCL_ALLGATHERV_SIMPLE_RING_SEGMENT_SIZE) } } },
        // Chunk size of a large call, in place of the CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE
        // spec and, for alltoall on Arc, of CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD (0 = no
        // knob). Bound to policy-only fields, which the consumers read before the spec.
        { "simple_chunk_size",
          knob_type::size,
          knob_phase::scaleup,
          { "allreduce", "reduce_scatter", "allgather", "alltoall" },
          {},
          {},
          { { "allreduce",
              bind_env<&knob_env::sycl_allreduce_simple_chunk_size_override>(
                  CCL_SYCL_ALLREDUCE_SIMPLE_CHUNK_SIZE) },
            { "reduce_scatter",
              bind_env<&knob_env::sycl_reduce_scatter_simple_chunk_size_override>(
                  CCL_SYCL_REDUCE_SCATTER_SIMPLE_CHUNK_SIZE) },
            { "allgather",
              bind_env<&knob_env::sycl_allgatherv_simple_chunk_size_override>(
                  CCL_SYCL_ALLGATHERV_SIMPLE_CHUNK_SIZE) },
            { "alltoall",
              bind_env<&knob_env::sycl_alltoall_simple_chunk_size_override>(
                  CCL_SYCL_ALLTOALL_SIMPLE_CHUNK_SIZE) } } },
        // Example, not a real knob: a per-collective binding. The policy line
        //   "algorithm=large": { "copy_engine=yes": {} }
        // sets knob_env::sycl_copy_engine for allreduce, and
        // knob_env::sycl_alltoall_copy_engine_mode for alltoall and alltoallv.
        //
        // { "copy_engine",
        //   knob_type::boolean,
        //   knob_phase::scaleup,
        //   { "allreduce", "alltoall", "alltoallv" },
        //   {},
        //   bind_env<&knob_env::sycl_copy_engine>(CCL_SYCL_COPY_ENGINE),
        //   { { "alltoall",
        //       bind_env<&knob_env::sycl_alltoall_copy_engine_mode>(
        //           CCL_SYCL_ALLTOALL_COPY_ENGINE) },
        //     { "alltoallv",
        //       bind_env<&knob_env::sycl_alltoall_copy_engine_mode>(
        //           CCL_SYCL_ALLTOALL_COPY_ENGINE) } } },
    };
    return knobs;
}

} // namespace selection
} // namespace ccl
