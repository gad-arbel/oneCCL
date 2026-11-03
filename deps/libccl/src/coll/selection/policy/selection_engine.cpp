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

#include "coll/selection/policy/selection_engine.hpp"

#include "coll/algorithms/algorithm_utils.hpp"  // ccl_coll_type_to_str
#include "comm/comm.hpp"
#include "common/env/vars_experimental.hpp"  // CCL_SELECTION_POLICY_PATH name
#include "common/global/global.hpp"          // ccl::global_data::env()
#include "common/log/log.hpp"
#include "common/utils/parse_utils.hpp"

#include "coll/selection/policy/mini_json.hpp"
#include "coll/selection/policy/policy_env.hpp"
#include "coll/selection/policy/policy_knobs.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace ccl {
namespace selection {

// -----------------------------------------------------------------------------
// Field / Predicate resolution — all string→enum work happens at load time.
// -----------------------------------------------------------------------------

namespace {

using Field = NodeKey::Field;
using Predicate = NodeKey::Predicate;
using Op = NodeKey::Op;

// Enum-value tables. Each maps a string spelling in the policy file to a
// small int. The engine never round-trips these back to strings on the hot
// path — value_enum_ is compared to a runtime enum promoted the same way.
struct EnumEntry { std::string_view name; int value; };

constexpr int kPhaseScaleup  = 0;
constexpr int kPhaseScaleout = 1;
constexpr EnumEntry kPhaseTable[] = {
    {"scaleup", kPhaseScaleup},
    {"scaleout", kPhaseScaleout},
};

// Values must match ccl_p2p_domain_level (see p2p_domain.hpp) so JSON
// `p2p_domain=<name>` compares against the runtime value with no
// translation table beyond the string→int here at load time.
constexpr EnumEntry kP2pDomainTable[] = {
    {"auto",   0},
    {"node",   1},
    {"numa",   2},
    {"switch", 3},
    {"none",   4},
};

// Values must match ccl_atl_transport (see env.hpp).
constexpr EnumEntry kTransportTable[] = {
    {"ofi", 0},
    {"mpi", 1},
};

// dtype table kept intentionally small — extend as JSON
// authors need them. Unknown values are Predicate::invalid (never matches)
// and are warned about at load time.
constexpr EnumEntry kDtypeTable[] = {
    {"int8", 0},  {"uint8", 1},
    {"int16", 2}, {"uint16", 3}, {"float16", 4}, {"bfloat16", 5},
    {"int32", 6}, {"uint32", 7}, {"float32", 8},
    {"int64", 9}, {"uint64", 10}, {"float64", 11},
};

std::optional<int> lookup_enum(const EnumEntry* table, std::size_t n,
                               std::string_view name) {
    for (std::size_t i = 0; i < n; ++i) {
        if (table[i].name == name) return table[i].value;
    }
    return std::nullopt;
}
#define LOOKUP(TABLE, NAME) lookup_enum(TABLE, sizeof(TABLE) / sizeof(TABLE[0]), NAME)

Field resolve_field(std::string_view name) {
    if (name == "msg_size")        return Field::msg_size;
    if (name == "send_msg_size")   return Field::send_msg_size;
    if (name == "count")           return Field::count;
    if (name == "comm_size")       return Field::comm_size;
    if (name == "phase")           return Field::phase;
    if (name == "p2p_domain")      return Field::p2p_domain;
    if (name == "dtype")           return Field::dtype;
    if (name == "has_p2p_access")  return Field::has_p2p_access;
    if (name == "is_multi_thread") return Field::is_multi_thread;
    if (name == "is_arc")          return Field::is_arc;
    if (name == "is_recording")    return Field::is_recording;
    if (name == "transport")       return Field::transport;
    if (name == "direct_gpu_rdma") return Field::direct_gpu_rdma;
    if (name == "use_copy_engine") return Field::use_copy_engine;
    if (name == "protocol")        return Field::protocol;
    if (name == "algorithm")       return Field::algorithm;
    return Field::unknown;
}

// A global_config key named after an env var ("CCL_SYCL_TMP_BUF_SIZE").
bool is_env_var_key(std::string_view key) {
    return key.size() > 4 && key.substr(0, 4) == "CCL_";
}

using utils::parse_bool;

bool is_power_of_two(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

// Promote the runtime dtype to the same small-int space kDtypeTable uses.
int runtime_dtype_enum(ccl_datatype dt) {
    switch (dt.idx()) {
        case ccl::datatype::int8:     return 0;
        case ccl::datatype::uint8:    return 1;
        case ccl::datatype::int16:    return 2;
        case ccl::datatype::uint16:   return 3;
        case ccl::datatype::float16:  return 4;
        case ccl::datatype::bfloat16: return 5;
        case ccl::datatype::int32:    return 6;
        case ccl::datatype::uint32:   return 7;
        case ccl::datatype::float32:  return 8;
        case ccl::datatype::int64:    return 9;
        case ccl::datatype::uint64:   return 10;
        case ccl::datatype::float64:  return 11;
        default:                      return -1;
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// NodeKey::parse — split raw "<param><op><value>" once, resolve field/enum
// values, and cache them. matches() then does no string work.
// -----------------------------------------------------------------------------

NodeKey NodeKey::parse(const std::string& raw) {
    NodeKey k;

    auto le = raw.find("<=");
    auto lt = raw.find('<');
    auto eq = raw.find('=');

    size_t op_pos = std::string::npos;
    size_t op_len = 0;

    if (le != std::string::npos) {
        op_pos = le; op_len = 2; k.op_ = Op::less_equal;
    }
    else if (lt != std::string::npos && (eq == std::string::npos || lt < eq)) {
        op_pos = lt; op_len = 1; k.op_ = Op::less;
    }
    else if (eq != std::string::npos) {
        op_pos = eq; op_len = 1; k.op_ = Op::equals;
    }

    if (op_pos == std::string::npos) {
        // No operator — treat as raw name; won't match anything.
        k.param_ = raw;
        k.field_ = Field::unknown;
        k.pred_ = Predicate::invalid;
        return k;
    }

    k.param_ = raw.substr(0, op_pos);
    k.raw_value_ = raw.substr(op_pos + op_len);
    k.field_ = resolve_field(k.param_);

    // Wildcard is universal — no per-field logic.
    if (k.raw_value_ == "any") {
        k.pred_ = Predicate::wildcard;
        return k;
    }

    switch (k.field_) {
        case Field::msg_size:
        case Field::send_msg_size:
        case Field::count:
        case Field::comm_size:
            if (k.raw_value_ == "pow2") {
                k.pred_ = Predicate::power_of_two;
            }
            else {
                // Shared number syntax (common/utils/parse_utils.hpp): digits with an
                // optional K/M/G/T (x1000) or Kb/Mb/Gb/Tb (x1024) suffix.
                const auto n = utils::parse_size(k.raw_value_);
                if (n.has_value()) {
                    k.pred_ = Predicate::integer;
                    k.value_int_ = *n;
                }
                else {
                    k.pred_ = Predicate::invalid;
                }
            }
            break;

        case Field::has_p2p_access:
        case Field::is_multi_thread:
        case Field::is_arc:
        case Field::is_recording:
        case Field::direct_gpu_rdma:
        case Field::use_copy_engine:
            if (auto b = parse_bool(k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::boolean : Predicate::invalid;
                k.value_bool_ = *b;
            }
            else {
                k.pred_ = Predicate::invalid;
            }
            break;

        case Field::phase:
            if (auto v = LOOKUP(kPhaseTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::p2p_domain:
            if (auto v = LOOKUP(kP2pDomainTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::transport:
            if (auto v = LOOKUP(kTransportTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::dtype:
            if (auto v = LOOKUP(kDtypeTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::protocol:
        case Field::algorithm:
            // Terminals — value stays as raw_value_; traverser consumes it.
            k.pred_ = Predicate::enum_val;  // sentinel; not evaluated
            break;

        case Field::unknown:
        case Field::leaf_param:
            k.pred_ = Predicate::invalid;
            break;
    }
    return k;
}

// -----------------------------------------------------------------------------
// NodeKey::matches — hot-path evaluator. No allocations, no string compares
// on the numeric/boolean/enum paths.
// -----------------------------------------------------------------------------

bool NodeKey::matches(const selection_engine_input& p) const {
    if (pred_ == Predicate::wildcard) return true;
    if (pred_ == Predicate::invalid)  return false;

    auto cmp_int = [&](size_t rt) {
        switch (op_) {
            case Op::equals:     return rt == value_int_;
            case Op::less:       return rt <  value_int_;
            case Op::less_equal: return rt <= value_int_;
        }
        return false;
    };

    auto cmp_opt_bool = [&](const std::optional<bool>& hint) {
        return hint.has_value() && (*hint == value_bool_);
    };

    switch (field_) {
        case Field::msg_size:
            return pred_ == Predicate::integer
                       ? cmp_int(p.msg_bytes)
                       : (op_ == Op::equals && is_power_of_two(p.msg_bytes));
        case Field::send_msg_size:
            if (!p.send_msg_bytes) return false;
            return pred_ == Predicate::integer
                       ? cmp_int(*p.send_msg_bytes)
                       : (op_ == Op::equals && is_power_of_two(*p.send_msg_bytes));
        case Field::count:
            return pred_ == Predicate::integer
                       ? cmp_int(p.count)
                       : (op_ == Op::equals && is_power_of_two(p.count));
        case Field::comm_size:
            return pred_ == Predicate::integer
                       ? cmp_int(p.comm_size)
                       : (op_ == Op::equals && is_power_of_two(p.comm_size));

        case Field::has_p2p_access:  return cmp_opt_bool(p.has_p2p_access);
        case Field::is_multi_thread: return cmp_opt_bool(p.is_multi_thread);
        case Field::is_arc:          return cmp_opt_bool(p.is_arc);
        case Field::is_recording:    return cmp_opt_bool(p.is_recording);
        case Field::direct_gpu_rdma: return cmp_opt_bool(p.direct_gpu_rdma);
        case Field::use_copy_engine: return cmp_opt_bool(p.use_copy_engine);

        case Field::phase:
            return p.phase.has_value() && static_cast<int>(*p.phase) == value_enum_;
        case Field::p2p_domain:
            return p.p2p_domain.has_value() && static_cast<int>(*p.p2p_domain) == value_enum_;
        case Field::transport:
            return p.transport.has_value() && static_cast<int>(*p.transport) == value_enum_;
        case Field::dtype: {
            int rt = runtime_dtype_enum(p.dtype);
            return rt >= 0 && rt == value_enum_;
        }

        case Field::protocol:
        case Field::algorithm:
        case Field::unknown:
        case Field::leaf_param:
            return false;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Tree construction from parsed JSON
// -----------------------------------------------------------------------------

namespace {

using json = mini_json::Value;

// Where in the tree build_node() is, for the knob checks.
struct build_context {
    const std::string& path;
    const std::string& collective; // policy root name, e.g. "allgather"
    knob_phase phase; // `any` until a phase= branch is seen
};

std::string join_names(const std::vector<std::string>& names) {
    std::string out;
    for (const auto& n : names) {
        out += (out.empty() ? "" : ", ") + n;
    }
    return out;
}

// Warns about a leaf knob the registry doesn't know, or that is invalid or
// misplaced. The knob is kept either way.
void check_leaf_knob(const build_context& ctx, const std::string& name, const std::string& value) {
    const auto* spec = find_leaf_knob(name);
    if (!spec) {
        std::vector<std::string> known;
        for (const auto& k : leaf_knobs()) {
            known.emplace_back(k.name);
        }
        LOG_WARN("selection policy: '",
                 ctx.path,
                 "': unknown knob '",
                 name,
                 "' under collective=",
                 ctx.collective,
                 "; nothing reads it. Known knobs: ",
                 join_names(known));
        return;
    }
    if (!is_valid_knob_value(spec->type, spec->choices, value)) {
        LOG_WARN("selection policy: '",
                 ctx.path,
                 "': invalid value '",
                 value,
                 "' for knob '",
                 name,
                 "' under collective=",
                 ctx.collective,
                 spec->choices.empty() ? "" : " (expected one of: ",
                 join_names(spec->choices),
                 spec->choices.empty() ? "" : ")");
    }
    if (!spec->collectives.empty() &&
        std::find(spec->collectives.begin(), spec->collectives.end(), ctx.collective) ==
            spec->collectives.end()) {
        LOG_WARN("selection policy: '",
                 ctx.path,
                 "': knob '",
                 name,
                 "' is not used by collective=",
                 ctx.collective,
                 " (only: ",
                 join_names(spec->collectives),
                 ")");
    }
    if (ctx.phase != knob_phase::any &&
        (static_cast<std::uint8_t>(spec->phase) & static_cast<std::uint8_t>(ctx.phase)) == 0) {
        LOG_WARN("selection policy: '",
                 ctx.path,
                 "': knob '",
                 name,
                 "' is not used in the ",
                 ctx.phase == knob_phase::scaleup ? "scaleup" : "scaleout",
                 " phase");
    }
}

std::unique_ptr<SelectionNode> build_node(const json& obj, const build_context& ctx) {
    const std::string& path = ctx.path;
    auto node = std::make_unique<SelectionNode>();
    if (!obj.is_object()) return node;

    std::unordered_set<std::string> seen_keys;

    for (auto it = obj.begin(); it != obj.end(); ++it) {
        // Author comments — keys starting with '_' — never take part in
        // dispatch and don't get flagged as unknown.
        if (!it.key().empty() && it.key()[0] == '_') continue;

        if (!seen_keys.insert(it.key()).second) {
            LOG_WARN("selection policy: '", path,
                     "': duplicate branch key '", it.key(),
                     "' at same level — later value wins");
        }

        NodeKey k = NodeKey::parse(it.key());

        // Leaf-scoped algorithm knobs (e.g. `"k=2": {}`): only classify as a
        // leaf when the key is NOT a recognized filter/terminal and body is
        // empty. Anything else is a branch.
        const bool is_leaf_knob =
            k.field() == Field::unknown &&
            k.op() == NodeKey::Op::equals && !k.raw_value().empty() &&
            it->is_object() && it->empty();

        if (is_leaf_knob) {
            check_leaf_knob(ctx, k.param(), k.raw_value());
            node->leaf_params_[k.param()] = k.raw_value();
            continue;
        }

        if (k.field() == Field::unknown) {
            LOG_WARN("selection policy: '", path,
                     "': unknown filter '", k.param(),
                     "' — this branch will never match; check spelling. "
                     "Known filters: msg_size, send_msg_size, count, comm_size, phase, "
                     "p2p_domain, dtype, has_p2p_access, "
                     "is_multi_thread, is_arc, is_recording, transport, direct_gpu_rdma, "
                     "use_copy_engine");
        }
        else if (k.predicate() == NodeKey::Predicate::invalid &&
                 k.field() != Field::protocol && k.field() != Field::algorithm) {
            LOG_WARN("selection policy: '", path,
                     "': branch '", it.key(),
                     "' has an unparseable value; this branch will never match");
        }

        build_context child_ctx = ctx;
        if (k.field() == Field::phase && k.op() == NodeKey::Op::equals) {
            if (k.raw_value() == "scaleup")
                child_ctx.phase = knob_phase::scaleup;
            if (k.raw_value() == "scaleout")
                child_ctx.phase = knob_phase::scaleout;
        }
        node->children_.emplace_back(std::move(k), build_node(*it, child_ctx));
    }
    return node;
}

}  // namespace

// -----------------------------------------------------------------------------
// SelectionEngine
// -----------------------------------------------------------------------------

void SelectionEngine::clear() {
    roots_.clear();
    global_config_.clear();
    loaded_ = false;
}

bool SelectionEngine::load_from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        LOG_WARN("selection policy: cannot open '", path, "'");
        return false;
    }
    json doc;
    try {
        doc = json::parse(in);
    }
    catch (const std::exception& e) {
        LOG_ERROR("selection policy: parse error in '", path, "': ", e.what());
        return false;
    }
    if (!doc.is_object()) {
        LOG_ERROR("selection policy: root of '", path, "' must be an object");
        return false;
    }

    roots_.clear();
    global_config_.clear();

    // Enforce schema_version match. Silent load of a stale schema is the
    // fastest way to mis-dispatch after a schema change.
    static constexpr long long kSchemaVersion = 1;
    if (const auto* sv = doc.find("schema_version")) {
        if (!sv->is_number_integer() || sv->get<long long>() != kSchemaVersion) {
            LOG_ERROR("selection policy: '", path,
                      "' has unsupported schema_version (expected ",
                      kSchemaVersion, ")");
            return false;
        }
    }
    else {
        LOG_WARN("selection policy: '", path,
                 "' has no schema_version; assuming ", kSchemaVersion);
    }

    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (it.key() == "schema_version") continue;

        if (it.key() == "global_config" && it->is_object()) {
            for (auto cf = it->begin(); cf != it->end(); ++cf) {
                // Author comments, as in the tree.
                if (!cf.key().empty() && cf.key()[0] == '_')
                    continue;
                if (cf->is_string()) {
                    global_config_[cf.key()] = cf->get<std::string>();
                }
                else if (cf->is_boolean()) {
                    // env vars spell booleans 1/0
                    global_config_[cf.key()] = cf->get<bool>() ? "1" : "0";
                }
                else if (cf->is_number_integer()) {
                    global_config_[cf.key()] = std::to_string(cf->get<long long>());
                }
                else if (cf->is_number_float()) {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%.17g", cf->get<double>());
                    global_config_[cf.key()] = buf;
                }
                else {
                    global_config_[cf.key()] = cf->dump();
                }
                // CCL_* keys are checked by the env parser, which knows the variables.
                if (!is_env_var_key(cf.key())) {
                    LOG_WARN("selection policy: '",
                             path,
                             "': unknown global_config key '",
                             cf.key(),
                             "'; nothing reads it. Expected a CCL_* environment variable");
                }
            }
            continue;
        }

        NodeKey k = NodeKey::parse(it.key());
        if (k.param() != "collective") {
            LOG_WARN("selection policy: '", path,
                     "': ignoring unknown top-level key '", it.key(),
                     "' (expected 'schema_version', 'global_config', or "
                     "'collective=<name>')");
            continue;
        }
        if (roots_.count(k.raw_value()) != 0) {
            LOG_WARN("selection policy: '", path,
                     "': duplicate 'collective=", k.raw_value(),
                     "' root — later definition overrides earlier");
        }
        roots_[k.raw_value()] =
            build_node(*it, build_context{ path, k.raw_value(), knob_phase::any });
    }
    loaded_ = !roots_.empty() || !global_config_.empty();
    LOG_INFO("selection policy: loaded ", roots_.size(),
             " collective(s) from '", path, "'");
    return true;
}

bool SelectionEngine::has_rules_for(const std::string& collective) const {
    if (roots_.empty()) return false;
    return roots_.count(collective) != 0;
}

std::unordered_map<std::string, std::string> SelectionEngine::env_overrides() const {
    std::unordered_map<std::string, std::string> out;
    if (!loaded_)
        return out;
    for (const auto& [key, value] : global_config_) {
        if (is_env_var_key(key))
            out.emplace(key, value);
    }
    return out;
}

// -----------------------------------------------------------------------------
// Traversal — backtracking DFS. A subtree that matches an interior filter but
// yields no terminal below no longer aborts the whole resolve; the caller can
// try later siblings (typically `<param>=any`). See CR §2.
// -----------------------------------------------------------------------------

namespace {

std::optional<LeafDecision> traverse(const SelectionNode* node,
                                     const selection_engine_input& p,
                                     std::optional<std::string> protocol_ctx) {
    if (!node) return std::nullopt;

    for (const auto& [key, child] : node->children_) {
        if (key.field() == Field::algorithm && key.op() == Op::equals) {
            LeafDecision d;
            d.algorithm = key.raw_value();
            d.protocol = std::move(protocol_ctx);
            if (child) d.params = child->leaf_params_;
            return d;
        }
        if (key.field() == Field::protocol && key.op() == Op::equals) {
            auto res = traverse(child.get(), p, key.raw_value());
            if (res.has_value()) return res;
            continue;
        }

        if (!key.matches(p)) continue;
        auto res = traverse(child.get(), p, protocol_ctx);
        if (res.has_value()) return res;
        // Interior match with no terminal below — keep trying later siblings
        // instead of committing to this branch.
    }
    return std::nullopt;
}

} // namespace

const char* ctype_to_policy_key(ccl_coll_type ctype) {
    // allgatherv shares the allgather policy tree — see collectives_policy.json.
    if (ctype == ccl_coll_allgatherv) return "allgather";
    return ccl_coll_type_to_str(ctype);
}

std::optional<LeafDecision> SelectionEngine::resolve(
    const std::string& collective, const selection_engine_input& input) const {
    auto it = roots_.find(collective);
    if (it == roots_.end()) return std::nullopt;
    return traverse(it->second.get(), input, std::nullopt);
}

std::optional<LeafDecision> SelectionEngine::resolve(
    ccl_coll_type ctype, const selection_engine_input& input) const {
    return resolve(std::string(ctype_to_policy_key(ctype)), input);
}

// -----------------------------------------------------------------------------
// Singleton (lazy, once-only load)
// -----------------------------------------------------------------------------

namespace {

// Resolve CCL_SELECTION_POLICY_PATH into a file path, or nullopt if unset.
std::optional<std::string> resolve_policy_path() {
    const auto& envval = ccl::global_data::env().selection_policy_path;
    if (envval.empty()) {
        return std::nullopt;
    }
    return envval;
}

}  // namespace

namespace {

SelectionEngine& engine_instance() {
    static SelectionEngine engine;
    return engine;
}

// Loads the policy once per process, from the first caller's path.
void load_engine_once(const std::optional<std::string>& path) {
    static std::once_flag once;
    std::call_once(once, [&]() {
        if (path) {
            engine_instance().load_from_file(*path);
        }
    });
}

} // namespace

SelectionEngine& global_engine() {
    load_engine_once(resolve_policy_path());
    return engine_instance();
}

std::unordered_map<std::string, std::string> policy_env_overrides(const std::string& path) {
    load_engine_once(path.empty() ? std::nullopt : std::optional<std::string>(path));
    return engine_instance().env_overrides();
}

}  // namespace selection
}  // namespace ccl
