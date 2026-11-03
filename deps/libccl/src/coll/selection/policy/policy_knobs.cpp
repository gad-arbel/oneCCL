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

#include "coll/selection/policy/policy_knobs.hpp"

#include <algorithm>
#include <cstdlib>

#include "coll/selection/policy/policy_env.hpp"
#include "coll/selection/policy/selection_engine.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace ccl {
namespace selection {

const leaf_knob_spec* find_leaf_knob(std::string_view name) {
    for (const auto& k : leaf_knobs()) {
        if (name == k.name) {
            return &k;
        }
    }
    return nullptr;
}

bool is_valid_knob_value(knob_type type,
                         const std::vector<std::string>& choices,
                         const std::string& value) {
    switch (type) {
        case knob_type::string: return !value.empty();
        case knob_type::boolean: return utils::parse_bool(value).has_value();
        case knob_type::integer: return utils::parse_int(value).has_value();
        case knob_type::size: return utils::parse_size(value).has_value();
        case knob_type::real: return utils::parse_double(value).has_value();
        case knob_type::choice: {
            const std::string v = knob_detail::lower(value);
            return std::find(choices.begin(), choices.end(), v) != choices.end();
        }
    }
    return false;
}

bool is_env_set(const char* cvar) {
    return cvar && std::getenv(cvar) != nullptr;
}

// -----------------------------------------------------------------------------
// Per-call env-bound leaf knobs
// -----------------------------------------------------------------------------

namespace {

bool in_phase(knob_phase spec, knob_phase phase) {
    return (static_cast<std::uint8_t>(spec) & static_cast<std::uint8_t>(phase)) != 0;
}

bool applies_to(const leaf_knob_spec& spec, const std::string& collective) {
    return spec.collectives.empty() ||
           std::find(spec.collectives.begin(), spec.collectives.end(), collective) !=
               spec.collectives.end();
}

// The binding of `spec` for `collective`: its env_per_collective entry, else
// `env` (empty for a value-only knob).
const env_binding& binding_for(const leaf_knob_spec& spec, const std::string& collective) {
    const auto it = spec.env_per_collective.find(collective);
    return it != spec.env_per_collective.end() ? it->second : spec.env;
}

// One entry per binding: a knob with per-collective bindings has several.
struct bound_knob {
    const leaf_knob_spec* spec;
    const env_binding* binding;
    std::string startup; // value before any policy write, restored before each consult
};

// Policy root names ("allgather", not "allgatherv"): the keys env_per_collective
// is matched against.
bool is_policy_collective(const std::string& name) {
    for (int c = 0; c < static_cast<int>(ccl_coll_last_value); ++c) {
        if (name == ctype_to_policy_key(static_cast<ccl_coll_type>(c))) {
            return true;
        }
    }
    return false;
}

// Warns once about env_per_collective entries that can never apply.
void check_registry() {
    static const bool checked = [] {
        for (const auto& k : leaf_knobs()) {
            for (const auto& [collective, binding] : k.env_per_collective) {
                if (!is_policy_collective(collective)) {
                    LOG_WARN("selection policy: knob ",
                             k.name,
                             ": env_per_collective key '",
                             collective,
                             "' is not a policy collective name (e.g. allgatherv is "
                             "'allgather'); this binding never applies");
                }
                else if (!applies_to(k, collective)) {
                    LOG_WARN("selection policy: knob ",
                             k.name,
                             ": env_per_collective key '",
                             collective,
                             "' is not in the knob's collectives; this binding "
                             "never applies");
                }
            }
        }
        return true;
    }();
    (void)checked;
}

// Captured on the first call for `phase`, i.e. after env parsing.
const std::vector<bound_knob>& bound_knobs(knob_phase phase) {
    check_registry();
    auto build = [phase] {
        std::vector<bound_knob> out;
        const auto& env = ccl::global_data::env().knobs;
        for (const auto& k : leaf_knobs()) {
            if (!in_phase(k.phase, phase)) {
                continue;
            }
            if (k.env) {
                out.push_back({ &k, &k.env, k.env.get(env) });
            }
            for (const auto& [collective, b] : k.env_per_collective) {
                if (b) {
                    out.push_back({ &k, &b, b.get(env) });
                }
            }
        }
        return out;
    };
    if (phase == knob_phase::scaleout) {
        static const std::vector<bound_knob> scaleout = build();
        return scaleout;
    }
    static const std::vector<bound_knob> scaleup = build();
    return scaleup;
}

// Set for the whole process once a multi-thread communicator exists
// (mt_comm.cpp). The knobs are off then.
bool is_multi_thread_process() {
    const auto& shared = ccl::global_data::get().shared_data;
    return shared && shared->is_multi_thread_instance;
}

// The knobs write this thread's knob_env, so calls on other threads keep their own values.
// `active` once a lookup ran on this thread. `knobs_live` is the multi-thread state as
// restore_policy_knobs() last saw it: holding it here keeps every call_env() read between
// two lookups on the same side, where re-reading the flag could answer differently halfway
// through a phase and give two ranks different knobs. A phase runs to the end before the
// next lookup, so that is every read of the knobs it applied.
struct thread_call_env {
    knob_env knobs;
    bool active = false;
    bool knobs_live = false;
};

thread_call_env& this_thread_call_env() {
    thread_local thread_call_env state;
    return state;
}

} // namespace

const knob_env& call_env() {
    const auto& state = this_thread_call_env();
    return state.knobs_live ? state.knobs : ccl::global_data::env().knobs;
}

void restore_policy_knobs(knob_phase phase) {
    auto& state = this_thread_call_env();
    state.knobs_live = !is_multi_thread_process();
    if (!state.knobs_live) {
        return; // env_data is shared by all threads: never written then
    }
    // the first lookup on a thread sets the fields of both phases, so all bound fields are valid
    for (const auto p : { knob_phase::scaleup, knob_phase::scaleout }) {
        if (p == phase || !state.active) {
            for (const auto& b : bound_knobs(p)) {
                b.binding->set(state.knobs, b.startup);
            }
        }
    }
    state.active = true;
}

std::string apply_policy_knobs(const LeafDecision& decision,
                               knob_phase phase,
                               const std::string& collective) {
    auto& state = this_thread_call_env();
    if (!state.knobs_live) {
        return {};
    }
    auto& knobs = state.knobs;
    std::string applied;
    for (const auto& b : bound_knobs(phase)) {
        const auto it = decision.params.find(b.spec->name);
        if (it == decision.params.end() || !applies_to(*b.spec, collective)) {
            continue; // the engine warned about misplaced knobs at load
        }
        if (b.binding != &binding_for(*b.spec, collective)) {
            continue; // another collective's binding of this knob
        }
        if (!b.binding->set(knobs, it->second)) {
            continue; // invalid values were warned about at load
        }
        applied += " " + it->first + "=" + it->second + " (" + b.binding->cvar + ")";
    }
    return applied;
}

} // namespace selection
} // namespace ccl
