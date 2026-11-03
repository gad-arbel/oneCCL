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

#include "coll/algorithms/utils/sycl_policy_resolver_json.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

#include "coll/selection/policy/policy_knobs.hpp"
#include "coll/selection/policy/selection_engine.hpp"
#include "coll/selection/selector_impl.hpp"  // fill_table_from_str
#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace ccl::sycl_policy_resolver::json {

using selection::selection_engine_input;
using selection::LeafDecision;
using selection::global_engine;

namespace {

// -----------------------------------------------------------------------------
// String → typed-enum decoders.
// -----------------------------------------------------------------------------

std::optional<sycl_policy::scaleup_protocol> decode_scaleup_protocol(std::string_view s) {
    if (s == "sycl_ll")     return sycl_policy::scaleup_protocol::ll;
    if (s == "sycl_simple") return sycl_policy::scaleup_protocol::default_kernel;
    if (s == "copy_engine") return sycl_policy::scaleup_protocol::copy_engine;
    return std::nullopt;
}

std::optional<sycl_policy::scaleup_algo> decode_scaleup_algo(std::string_view s) {
    if (s == "ring")     return sycl_policy::scaleup_algo::ring;
    if (s == "oneshot")  return sycl_policy::scaleup_algo::oneshot;
    if (s == "twoshots") return sycl_policy::scaleup_algo::twoshots;
    if (s == "small")    return sycl_policy::scaleup_algo::small;
    if (s == "large")    return sycl_policy::scaleup_algo::large;
    if (s == "sycl_copy_engine") return sycl_policy::scaleup_algo::sycl_copy_engine;
    if (s == "l0_copy_engine")   return sycl_policy::scaleup_algo::l0_copy_engine;
    return std::nullopt;
}

std::optional<sycl_policy::scaleout_algo> decode_scaleout_algo(std::string_view s) {
    if (s == "ring")               return sycl_policy::scaleout_algo::ring;
    if (s == "direct")             return sycl_policy::scaleout_algo::direct;
    if (s == "rabenseifner")       return sycl_policy::scaleout_algo::rabenseifner;
    if (s == "scatter_two_phase")  return sycl_policy::scaleout_algo::scatter_two_phase;
    if (s == "gdr_only_pairwise")  return sycl_policy::scaleout_algo::gdr_only_pairwise;
    if (s == "scatter")            return sycl_policy::scaleout_algo::scatter;
    if (s == "numa_gdr_split")     return sycl_policy::scaleout_algo::numa_gdr_split;
    if (s == "non_sycl")           return sycl_policy::scaleout_algo::non_sycl;
    if (s == "scatter_allgather")  return sycl_policy::scaleout_algo::scatter_allgather;
    return std::nullopt;
}

// Parses an `ll_table` knob ("<algo>:<lo>-<hi>;...", same format as
// CCL_SYCL_*_LL) once and caches it for the process lifetime, since
// `ll_algo_lookup` keeps a pointer to it. Returns nullptr if the spec is
// invalid, names a non-LL algorithm, or does not cover every size.
const ccl_selection_table_t<std::string>* get_cached_ll_table(const std::string& spec) {
    static std::shared_mutex mutex;
    static std::unordered_map<std::string, std::optional<ccl_selection_table_t<std::string>>> cache;

    // Fast path: every call after the first for a spec only reads.
    {
        std::shared_lock<std::shared_mutex> lock(mutex);
        const auto it = cache.find(spec);
        if (it != cache.end()) return it->second ? &*it->second : nullptr;
    }

    std::unique_lock<std::shared_mutex> lock(mutex);
    auto [it, inserted] = cache.try_emplace(spec);
    if (inserted) {
        ccl_selection_table_t<std::string> table;
        try {
            fill_table_from_str(spec, table);
        }
        catch (const std::exception& e) {
            LOG_WARN("scaleup json: invalid ll_table='", spec, "': ", e.what());
            return nullptr;
        }
        if (table.find(CCL_SELECTION_MAX_COLL_SIZE) == table.end()) {
            LOG_WARN("scaleup json: ll_table='", spec, "' does not cover sizes up to max");
            return nullptr;
        }
        for (const auto& [size, entry] : table) {
            const auto algo = decode_scaleup_algo(entry.first);
            if (!algo || sycl_policy::is_size_algo(*algo) ||
                sycl_policy::is_copy_engine_algo(*algo)) {
                LOG_WARN("scaleup json: ll_table='", spec, "' has non-LL algo '", entry.first,
                         "'");
                return nullptr;
            }
        }
        it->second = std::move(table);
    }
    return it->second ? &*it->second : nullptr;
}

// Pipeline chunk size for a JSON scaleout decision, using the same rules as
// the built-in selectors.
std::size_t resolve_scaleout_chunk_bytes(const sycl_policy::policy_query& q,
                                         sycl_policy::scaleout_algo algo,
                                         const LeafDecision& decision) {
    const auto& env = ccl::global_data::env();
    const std::size_t max_chunk = env.sycl_max_pipeline_chunk_size;

    // Only pipelined algorithms get a tuned chunk; the C++ allgatherv selector
    // tunes it only over MPI.
    using sycl_policy::scaleout_algo;
    const bool pipelined = algo == scaleout_algo::ring ||
                           algo == scaleout_algo::rabenseifner;
    const bool tuned =
        pipelined && (q.ctype != ccl_coll_allgatherv || env.atl_transport == ccl_atl_mpi);

    // the knob wins over CCL_SYCL_PIPELINE_CHUNK_SIZE
    const auto text = decision.knob("chunk_bytes");
    if (text.has_value()) {
        const std::size_t align = sycl_policy::pipeline_alignment(q.dtype.size());
        const auto bytes = utils::parse_size(*text);
        if (bytes.has_value() && *bytes >= align) {
            return std::min(static_cast<std::size_t>(*bytes), max_chunk);
        }
        LOG_WARN_ONCE("scaleout json: invalid chunk_bytes='",
                      *text,
                      "' (must be a size >= ",
                      align,
                      "), ignored; ");
    }

    if (tuned && env.sycl_pipeline_chunk_size != CCL_ENV_SIZET_NOT_SPECIFIED) {
        return std::min(static_cast<std::size_t>(env.sycl_pipeline_chunk_size), max_chunk);
    }

    if (!tuned) {
        return sycl_policy::default_pipeline_chunk_size;
    }
    switch (q.ctype) {
        case ccl_coll_allgatherv:
            return sycl_scaleout_selector::allgatherv_chunk_size(algo, q.msg_bytes, q.r2r_comm_size);
        case ccl_coll_allreduce:
            return sycl_scaleout_selector::allreduce_chunk_size(algo, q.msg_bytes, q.r2r_comm_size);
        case ccl_coll_reduce_scatter:
            return sycl_scaleout_selector::reduce_scatter_chunk_size(algo, q.msg_bytes, q.r2r_comm_size);
        default:
            return sycl_policy::default_pipeline_chunk_size;
    }
}

// Applies the env-bound knobs of `phase` from `decision`, even if its
// protocol/algorithm is later dropped.
void apply_policy_knobs(const sycl_policy::policy_query& q,
                        selection::knob_phase phase,
                        const LeafDecision* decision) {
    if (!decision) {
        return;
    }
    const std::string coll = selection::ctype_to_policy_key(q.ctype);
    const auto applied = selection::apply_policy_knobs(*decision, phase, coll);
    if (!applied.empty()) {
        LOG_INFO(phase == selection::knob_phase::scaleup ? "scaleup" : "scaleout",
                 " json: ",
                 coll,
                 " knobs:",
                 applied);
    }
}

// The policy lookup for one phase; nullopt without a policy or a matching rule.
std::optional<LeafDecision> lookup(const sycl_policy::policy_query& q,
                                   const selection::SelectionEngine& engine,
                                   selection_engine_input::phase_t phase,
                                   const char* phase_name) {
    auto decision = engine.resolve(q.ctype, q.to_selection_engine_input(phase));
    if (!decision) {
        LOG_DEBUG(
            phase_name, " json: no matching rule for ctype=", q.ctype, " msg_bytes=", q.msg_bytes);
    }
    return decision;
}

// The policy decision for one phase, with its env-bound leaf knobs applied.
// One lookup per call, for both the knobs and the decision.
std::optional<LeafDecision> lookup_with_knobs(const sycl_policy::policy_query& q,
                                              const selection::SelectionEngine& engine,
                                              selection_engine_input::phase_t phase,
                                              const char* phase_name) {
    const auto knobs = phase == selection_engine_input::phase_t::scaleup
                           ? selection::knob_phase::scaleup
                           : selection::knob_phase::scaleout;
    selection::restore_policy_knobs(knobs);
    auto decision = lookup(q, engine, phase, phase_name);
    apply_policy_knobs(q, knobs, decision ? &*decision : nullptr);
    return decision;
}

}  // namespace

// -----------------------------------------------------------------------------
// scaleup
// -----------------------------------------------------------------------------
std::optional<sycl_policy::sycl_scaleup_tune_attr> scaleup(
    const sycl_policy::policy_query& q) {
    return scaleup(q, global_engine());
}

std::optional<sycl_policy::sycl_scaleup_tune_attr> scaleup(
    const sycl_policy::policy_query& q,
    const selection::SelectionEngine& engine) {
    if (!engine.is_loaded()) {
        LOG_DEBUG("scaleup json: engine not loaded, falling back to C++ resolver");
        return std::nullopt;
    }
    const auto decision =
        lookup_with_knobs(q, engine, selection_engine_input::phase_t::scaleup, "scaleup");
    if (!decision)
        return std::nullopt;

    // Scaleup requires both a protocol and an algorithm.
    if (!decision->protocol) {
        LOG_WARN_ONCE("scaleup json: decision missing protocol for ctype=", q.ctype,
                      " (algo='", decision->algorithm, "'), dropping; ");
        return std::nullopt;
    }
    auto proto = decode_scaleup_protocol(*decision->protocol);
    if (!proto) {
        LOG_WARN_ONCE("scaleup json: unknown protocol '", *decision->protocol, "', dropping; ");
        return std::nullopt;
    }
    auto algo = decode_scaleup_algo(decision->algorithm);
    if (!algo) {
        LOG_WARN_ONCE("scaleup json: unknown algo '", decision->algorithm, "', dropping; ");
        return std::nullopt;
    }

    // LL is Arc-only.
    if (*proto == sycl_policy::scaleup_protocol::ll && !q.is_arc()) {
        LOG_DEBUG("scaleup json: LL requested on non-Arc, falling back");
        return std::nullopt;
    }

    // The copy engine is opt-in via CCL_ENABLE_ZE_CE and Arc-only, like the C++ path.
    if (*proto == sycl_policy::scaleup_protocol::copy_engine) {
        if (!ccl::global_data::env().enable_ze_ce) {
            LOG_DEBUG("scaleup json: copy engine requested but CCL_ENABLE_ZE_CE=0, falling back");
            return std::nullopt;
        }
        if (!q.is_arc()) {
            LOG_DEBUG("scaleup json: copy engine requested on non-Arc, falling back");
            return std::nullopt;
        }
    }

    // sycl_ll takes an LL algorithm, copy_engine a copy engine algorithm, the others a size algo.
    const bool is_ll = *proto == sycl_policy::scaleup_protocol::ll;
    const bool is_ce = *proto == sycl_policy::scaleup_protocol::copy_engine;
    const bool algo_fits = is_ce   ? sycl_policy::is_copy_engine_algo(*algo)
                           : is_ll ? !sycl_policy::is_size_algo(*algo) &&
                                         !sycl_policy::is_copy_engine_algo(*algo)
                                   : sycl_policy::is_size_algo(*algo);
    if (!algo_fits) {
        LOG_WARN_ONCE("scaleup json: algo '", decision->algorithm, "' does not fit protocol '",
                      *decision->protocol, "', dropping; ");
        return std::nullopt;
    }

    // The LL dispatcher reads the per-chunk algorithm from `tune.ll`: from the
    // `ll_table` knob when present, otherwise `algorithm` for every chunk.
    sycl_policy::ll_algo_lookup ll;
    if (is_ll) {
        const auto spec = decision->knob("ll_table");
        if (!spec.has_value()) {
            ll = sycl_policy::ll_algo_lookup::fixed(*algo);
        }
        else {
            const auto* table = get_cached_ll_table(*spec);
            if (table == nullptr) {
                LOG_WARN_ONCE("scaleup json: ll_table='", *spec, "' is invalid, dropping; ");
                return std::nullopt;
            }
            ll = sycl_policy::ll_algo_lookup{ *table };
        }
    }

    LOG_DEBUG("scaleup json: hit protocol='", *decision->protocol,
              "' algo='", decision->algorithm, "'");
    return sycl_policy::sycl_scaleup_tune_attr{ *proto, *algo, ll };
}

// -----------------------------------------------------------------------------
// scaleout
// -----------------------------------------------------------------------------
std::optional<::sycl_scaleout_tune_attr> scaleout(
    const sycl_policy::policy_query& q) {
    return scaleout(q, global_engine());
}

std::optional<::sycl_scaleout_tune_attr> scaleout(
    const sycl_policy::policy_query& q,
    const selection::SelectionEngine& engine) {
    if (!engine.is_loaded()) {
        LOG_DEBUG("scaleout json: engine not loaded, falling back to C++ resolver");
        return std::nullopt;
    }
    const auto decision =
        lookup_with_knobs(q, engine, selection_engine_input::phase_t::scaleout, "scaleout");
    if (!decision)
        return std::nullopt;

    auto algo = decode_scaleout_algo(decision->algorithm);
    if (!algo) {
        LOG_WARN_ONCE("scaleout json: unknown algo '", decision->algorithm, "', dropping; ");
        return std::nullopt;
    }

    ::sycl_scaleout_tune_attr out{ *algo, resolve_scaleout_chunk_bytes(q, *algo, *decision) };
    LOG_DEBUG("scaleout json: hit algo='", decision->algorithm,
              "' chunk_bytes=", out.pipeline_chunk_size());
    return out;
}

}  // namespace ccl::sycl_policy_resolver::json
