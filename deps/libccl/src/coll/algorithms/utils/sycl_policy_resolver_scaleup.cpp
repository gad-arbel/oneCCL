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

#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"

#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_json.hpp"
#include "coll/selection/selector.hpp"  // fill_table_from_str, via selector_impl.hpp
#include "common/log/log.hpp"
#include "comm/comm.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <optional>

namespace ccl::sycl_policy_resolver {

using sycl_policy::scaleup_algo;
using sycl_policy::scaleup_protocol;
using sycl_policy::sycl_scaleup_tune_attr;
using sycl_policy::ll_algo_lookup;
using sycl_policy::pick_size_algo;

namespace {

// Logs the resolver's decision. `source` is "" for the C++ cascade and
// "[json]" for a policy decision.
void log_tune(const sycl_policy::policy_query& query,
              const char* coll,
              const char* source,
              const sycl_scaleup_tune_attr& tune) {
    LOG_DEBUG("scaleup ", coll, source,
              " protocol=",        sycl_policy::to_str(tune.protocol),
              " algo=",            sycl_policy::to_str(tune.algo),
              " msg_bytes=",       query.msg_bytes,
              " count=",           query.count,
              " send_bytes=",      query.scaleup_send_bytes(),
              " is_arc=",          query.is_arc(),
              " is_multi_thread=", query.is_multi_thread(),
              " has_p2p_access=",  query.has_p2p_access(),
              " is_recording=",    query.is_recording);
}

// -----------------------------------------------------------------------------
// Per-collective LL table, parsed once. Spec: "<algo>:<lo>-<hi>;..." or a
// single algorithm; empty means ring.
// -----------------------------------------------------------------------------
struct ll_allgatherv_tag {};
struct ll_allreduce_tag {};
struct ll_reduce_scatter_tag {};

template <typename Tag>
const ccl_selection_table_t<std::string>& get_ll_table(const std::string& spec) {
    static const ccl_selection_table_t<std::string> table = [&] {
        ccl_selection_table_t<std::string> t;
        if (!spec.empty()) fill_table_from_str(spec, t);
        return t;
    }();
    return table;
}

// -----------------------------------------------------------------------------
// Two-stage resolver: stage 1 picks the protocol, stage 2 the size algo.
// -----------------------------------------------------------------------------

// Stage 1 — pick the protocol. `gate_size` is in the unit of `simple_threshold`
// (bytes, or elements for alltoallv).
scaleup_protocol resolve_protocol(const sycl_policy::policy_query& q,
                                  std::size_t gate_size,
                                  std::size_t simple_threshold,
                                  bool threshold_active = true) {
    if (q.is_arc() &&
        (q.is_multi_thread() ||
         (threshold_active && gate_size <= simple_threshold))) {
        return scaleup_protocol::ll;
    }
    return scaleup_protocol::default_kernel;
}

// -----------------------------------------------------------------------------
// tmp buffer capacity: the default `small` kernel stages the whole message in
// the tmp buffer, so it can only run when msg_bytes fits.
// -----------------------------------------------------------------------------

// Unlimited when the query carries no comm to read the size from.
std::size_t tmp_buf_capacity_of(const ccl_comm* c) {
    return c ? c->get_tmp_buf_size() : std::numeric_limits<std::size_t>::max();
}

std::size_t node_tmp_buf_capacity_of(const ccl_comm* c) {
    return tmp_buf_capacity_of(c ? c->get_node_comm().get() : nullptr);
}

// A collective's size-algo thresholds and the tmp buffer its default `small` kernel stages into.
struct size_limits {
    std::size_t small_threshold  = std::numeric_limits<std::size_t>::max();
    std::size_t tmp_buf_capacity = std::numeric_limits<std::size_t>::max();
};

// Stage 2: small/large (default). The default kernel picks `small` only when
// the message also fits its tmp buffer.
scaleup_algo pick_size_algo_for_protocol(scaleup_protocol proto,
                                         std::size_t send_bytes,
                                         const size_limits& limits) {
    switch (proto) {
        case scaleup_protocol::ll:
            return scaleup_algo::unspecified;
        case scaleup_protocol::default_kernel:
            if (send_bytes <= limits.small_threshold && send_bytes > limits.tmp_buf_capacity) {
                LOG_DEBUG("scaleup small: send_bytes ", send_bytes, " > tmp_buf_size ",
                          limits.tmp_buf_capacity, ", using large");
                return scaleup_algo::large;
            }
            return pick_size_algo(send_bytes, limits.small_threshold);
        case scaleup_protocol::copy_engine:
            return scaleup_algo::l0_copy_engine;
    }
    return scaleup_algo::unspecified;
}

// The two-stage pick for allgatherv / reduce_scatter / allreduce.
template <typename LLTag>
sycl_scaleup_tune_attr resolve_size_collective(const sycl_policy::policy_query& q,
                                               std::size_t send_bytes,
                                               std::size_t simple_threshold,
                                               const size_limits& limits,
                                               const std::string& ll_spec) {
    const auto proto = resolve_protocol(q, send_bytes, simple_threshold);
    const auto algo  = pick_size_algo_for_protocol(proto, send_bytes, limits);
    // Parse the LL table only when the LL path will consume it.
    const auto ll = proto == scaleup_protocol::ll ? ll_algo_lookup{ get_ll_table<LLTag>(ll_spec) }
                                                  : ll_algo_lookup{};
    return sycl_scaleup_tune_attr{ proto, algo, ll };
}

// For collectives whose dispatcher has no size algo.
constexpr size_limits no_size_limits{};

// False only for a default-kernel `small` whose message overflows the tmp buffer.
bool fits_tmp_buf(const sycl_scaleup_tune_attr& tune,
                  std::size_t send_bytes,
                  const size_limits& limits) {
    return tune.protocol != scaleup_protocol::default_kernel || tune.algo != scaleup_algo::small ||
           send_bytes <= limits.tmp_buf_capacity;
}

// One (protocol, algo) pair a collective's scaleup dispatcher can run.
struct scaleup_choice {
    scaleup_protocol protocol;
    scaleup_algo algo;
};

// Whether the L0 copy engine broadcast can run. Every failed requirement warns, except a small
// payload, which is ordinary routing.
bool can_use_l0_copy_engine_broadcast(const sycl_policy::policy_query& query) {
    const auto& env = ccl::global_data::env();
    if (!env.enable_ze_ce || !query.comm || !query.p2p_comm || !query.global_stream) {
        return false;
    }

    if (query.msg_bytes <= env.sycl_broadcast_small_threshold) {
        LOG_INFO("Native Level Zero command-list broadcast is a large-payload path; "
                 "falling back to another broadcast implementation");
        return false;
    }

    // counter based events exist only on ARC
    if (!query.is_arc()) {
        LOG_WARN("Native Level Zero command-list broadcast requires an ARC card; "
                 "falling back to another broadcast implementation");
        return false;
    }

    if (query.global_stream->get_backend() != sycl::backend::ext_oneapi_level_zero) {
        LOG_WARN("Native Level Zero command-list broadcast is unavailable for this queue, "
                 "which is not on the Level Zero backend; "
                 "falling back to another broadcast implementation");
        return false;
    }

    // the CE queue is not captured into a SYCL graph
    if (query.is_recording) {
        LOG_WARN("Native Level Zero command-list broadcast is unavailable while the queue "
                 "records a SYCL graph; falling back to another broadcast implementation");
        return false;
    }

    if (env.sycl_broadcast_tmp_buf) {
        LOG_WARN("Native Level Zero command-list broadcast does not support "
                 "CCL_SYCL_BROADCAST_TMP_BUF=1; falling back to another broadcast implementation");
        return false;
    }

    // the CE flow syncs over the node comm, so the selected P2P domain must be the node
    if (query.p2p_comm != query.comm->get_node_comm().get()) {
        LOG_WARN("Native Level Zero command-list broadcast requires the P2P domain to span "
                 "the node; falling back to another broadcast implementation");
        return false;
    }

    // multi thread comms are forced onto the tmp buffer path
    if (query.is_multi_thread()) {
        LOG_WARN("Native Level Zero command-list broadcast does not support multi-thread "
                 "communicators; falling back to another broadcast implementation");
        return false;
    }

    // scale-up only: the scale-out phase leaves a host event the CE flow cannot wait on;
    // is_uniform mirrors the scale-up condition in broadcast_sycl()
    auto r2r_comm = query.comm->get_r2r_comm();
    if (r2r_comm->size() > 1 || !r2r_comm->is_uniform) {
        LOG_WARN("Native Level Zero command-list broadcast is unavailable if there is more "
                 "than one node, or if the ranks are distributed unevenly across nodes; "
                 "falling back to another broadcast implementation");
        return false;
    }

    return true;
}

// Returns the JSON decision if `coll` supports it and it can run, otherwise nullopt.
std::optional<sycl_scaleup_tune_attr> from_json(const sycl_policy::policy_query& query,
                                                const char* coll,
                                                std::initializer_list<scaleup_choice> supported,
                                                const size_limits& limits) {
    auto tune = json::scaleup(query);
    if (!tune) return tune;
    const std::size_t send_bytes = query.scaleup_send_bytes();
    if (tune->ll.table) {
        for (const auto& [size, entry] : *tune->ll.table) {
            const bool entry_ok =
                std::any_of(supported.begin(), supported.end(), [&](const scaleup_choice& c) {
                    return c.protocol == scaleup_protocol::ll &&
                           entry.first == sycl_policy::to_str(c.algo);
                });
            if (!entry_ok) {
                LOG_WARN_ONCE("scaleup ", coll, "[json] ll_table algo '", entry.first,
                              "' not supported, falling back; ");
                return std::nullopt;
            }
        }
        tune->algo = scaleup_algo::unspecified;
    }
    else {
        const bool ok =
            std::any_of(supported.begin(), supported.end(), [&](const scaleup_choice& c) {
                return c.protocol == tune->protocol && c.algo == tune->algo;
            });
        if (!ok) {
            LOG_WARN_ONCE("scaleup ", coll, "[json] protocol '",
                          sycl_policy::to_str(tune->protocol), "' algo '",
                          sycl_policy::to_str(tune->algo),
                          "' not supported, falling back; ");
            return std::nullopt;
        }
    }
    // Same rule as pick_size_algo_for_protocol(): a default-kernel `small` that
    // doesn't fit the tmp buffer runs as `large`. The small kernels don't check
    // this themselves.
    if (!fits_tmp_buf(*tune, send_bytes, limits)) {
        const bool large_ok =
            std::any_of(supported.begin(), supported.end(), [](const scaleup_choice& c) {
                return c.protocol == scaleup_protocol::default_kernel &&
                       c.algo == scaleup_algo::large;
            });
        if (!large_ok) {
            LOG_WARN_ONCE("scaleup ", coll,
                          "[json] algo 'small' overflows the tmp buffer and "
                          "'large' is not supported, falling back; ");
            return std::nullopt;
        }
        LOG_DEBUG("scaleup ", coll, "[json] small: send_bytes ", send_bytes,
                  " > tmp_buf_size ", limits.tmp_buf_capacity, ", using large");
        tune->algo = scaleup_algo::large;
    }
    log_tune(query, coll, "[json]", *tune);
    return tune;
}

}  // namespace

// -----------------------------------------------------------------------------
// allgatherv scaleup
//
// Arc + (multi-thread || small message || no p2p) → LL256; otherwise the
// default kernel.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::allgatherv() const {
    const auto& env = ccl::global_data::env();
    size_limits limits;
    limits.small_threshold  = env.sycl_allgatherv_small_threshold;
    limits.tmp_buf_capacity = node_tmp_buf_capacity_of(query.comm);
    if (auto json_hit = from_json(query, "allgatherv",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::ll, scaleup_algo::oneshot },
                                    { scaleup_protocol::default_kernel, scaleup_algo::small },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  limits)) {
        return *json_hit;
    }

    const auto tune = resolve_size_collective<ll_allgatherv_tag>(
        query, query.msg_bytes, env.sycl_allgatherv_simple_threshold, limits,
        env.sycl_allgatherv_ll_algo);
    log_tune(query, "allgatherv", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// reduce_scatter scaleup — same shape as allgatherv. The C++ gates and the tmp
// buffer check use the send-buffer size (recv_count × p2p ranks), so the query
// must carry `p2p_comm`.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::reduce_scatter() const {
    CCL_THROW_IF_NOT(query.p2p_comm,
                     "reduce_scatter scaleup query needs with_p2p_comm() for its send bytes");
    const auto& env = ccl::global_data::env();
    size_limits limits;
    limits.small_threshold  = env.sycl_reduce_scatter_small_threshold;
    limits.tmp_buf_capacity = tmp_buf_capacity_of(query.p2p_comm);
    if (auto json_hit = from_json(query, "reduce_scatter",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::ll, scaleup_algo::oneshot },
                                    { scaleup_protocol::default_kernel, scaleup_algo::small },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  limits)) {
        return *json_hit;
    }

    const auto tune = resolve_size_collective<ll_reduce_scatter_tag>(
        query, query.scaleup_send_bytes(), env.sycl_reduce_scatter_simple_threshold, limits,
        env.sycl_reduce_scatter_ll_algo);
    log_tune(query, "reduce_scatter", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// allreduce scaleup — same as allgatherv, plus twoshots.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::allreduce() const {
    const auto& env = ccl::global_data::env();
    size_limits limits;
    limits.small_threshold  = env.sycl_allreduce_small_threshold;
    limits.tmp_buf_capacity = tmp_buf_capacity_of(query.p2p_comm);
    if (auto json_hit = from_json(query, "allreduce",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::ll, scaleup_algo::oneshot },
                                    { scaleup_protocol::ll, scaleup_algo::twoshots },
                                    { scaleup_protocol::default_kernel, scaleup_algo::small },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  limits)) {
        return *json_hit;
    }

    const auto tune = resolve_size_collective<ll_allreduce_tag>(
        query, query.msg_bytes, env.sycl_allreduce_simple_threshold, limits,
        env.sycl_allreduce_ll_algo);
    log_tune(query, "allreduce", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// alltoall scaleup — only the protocol is used.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::alltoall() const {
    if (auto json_hit = from_json(query, "alltoall",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  no_size_limits)) {
        return *json_hit;
    }
    const auto proto = resolve_protocol(query, query.msg_bytes,
                                        ccl::global_data::env().sycl_alltoall_simple_threshold,
                                        /*threshold_active=*/true);
    sycl_scaleup_tune_attr tune{ proto, proto == scaleup_protocol::ll ? scaleup_algo::ring
                                                                      : scaleup_algo::large };
    log_tune(query, "alltoall", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// alltoallv scaleup — same as alltoall; the threshold is in elements.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::alltoallv() const {
    if (auto json_hit = from_json(query, "alltoallv",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  no_size_limits)) {
        return *json_hit;
    }
    const auto& env = ccl::global_data::env();
    const auto proto = resolve_protocol(query, query.count, env.sycl_alltoall_simple_threshold,
                                        /*threshold_active=*/env.sycl_alltoall_tmp_buf);
    sycl_scaleup_tune_attr tune{ proto, proto == scaleup_protocol::ll ? scaleup_algo::ring
                                                                      : scaleup_algo::large };
    log_tune(query, "alltoallv", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// broadcast scaleup — LL ring on Arc, except large messages on non-recorded single-thread comms,
// which take the copy engine with CCL_ENABLE_ZE_CE and the default kernel otherwise.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::broadcast() const {
    const auto& env = ccl::global_data::env();
    size_limits limits;
    limits.small_threshold  = env.sycl_broadcast_small_threshold;
    limits.tmp_buf_capacity = node_tmp_buf_capacity_of(query.p2p_comm);
    const bool use_l0_copy_engine = can_use_l0_copy_engine_broadcast(query);
    if (auto json_hit = from_json(query, "broadcast",
                                  { { scaleup_protocol::ll, scaleup_algo::ring },
                                    { scaleup_protocol::default_kernel, scaleup_algo::small },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large },
                                    { scaleup_protocol::copy_engine,
                                      scaleup_algo::sycl_copy_engine },
                                    { scaleup_protocol::copy_engine,
                                      scaleup_algo::l0_copy_engine } },
                                  limits)) {
        if (json_hit->algo != scaleup_algo::l0_copy_engine || use_l0_copy_engine) {
            return *json_hit;
        }
    }

    const sycl_scaleup_tune_attr default_tune{
        scaleup_protocol::default_kernel,
        pick_size_algo_for_protocol(scaleup_protocol::default_kernel, query.msg_bytes, limits)
    };
    if (!query.is_arc()) {
        log_tune(query, "broadcast", "", default_tune);
        return default_tune;
    }
    // The Arc gate uses the threshold alone; the tmp buffer only limits the default kernel.
    const bool is_large =
        pick_size_algo(query.msg_bytes, limits.small_threshold) == scaleup_algo::large;

    // CCL_ENABLE_ZE_CE opts large messages into the L0 copy engine ahead of the LL gate when
    // the CE flow can run; otherwise the regular selection below applies unchanged.
    // sycl_copy_engine is reachable from a JSON policy rule only.
    if (use_l0_copy_engine) {
        const sycl_scaleup_tune_attr ce_tune{ scaleup_protocol::copy_engine,
                                              scaleup_algo::l0_copy_engine };
        log_tune(query, "broadcast", "", ce_tune);
        return ce_tune;
    }

    const bool use_ring = env.sycl_broadcast_ring;
    LOG_DEBUG("scaleup broadcast[gate] is_large=", is_large,
              " use_ring=", use_ring,
              " is_recording=", query.is_recording);
    // Multi-thread comms take LL at every size, like the collectives that use resolve_protocol().
    const auto tune = (query.is_recording || query.is_multi_thread() || !use_ring || !is_large)
                          ? sycl_scaleup_tune_attr{ scaleup_protocol::ll, scaleup_algo::ring }
                          : default_tune;
    log_tune(query, "broadcast", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// scatter scaleup — default kernel only, small/large.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::scatter() const {
    size_limits limits;
    limits.small_threshold  = ccl::global_data::env().sycl_scatter_small_threshold;
    limits.tmp_buf_capacity = tmp_buf_capacity_of(query.p2p_comm);
    if (auto json_hit = from_json(query, "scatter",
                                  { { scaleup_protocol::default_kernel, scaleup_algo::small },
                                    { scaleup_protocol::default_kernel, scaleup_algo::large } },
                                  limits)) {
        return *json_hit;
    }

    const sycl_scaleup_tune_attr tune{
        scaleup_protocol::default_kernel,
        pick_size_algo_for_protocol(scaleup_protocol::default_kernel, query.msg_bytes, limits)
    };
    log_tune(query, "scatter", "", tune);
    return tune;
}

// -----------------------------------------------------------------------------
// reduce scaleup — Arc-only.
// -----------------------------------------------------------------------------
sycl_scaleup_tune_attr scaleup_resolver::reduce() const {
    if (auto json_hit = from_json(query, "reduce", { { scaleup_protocol::ll, scaleup_algo::ring } },
                                  no_size_limits)) {
        return *json_hit;
    }
    sycl_scaleup_tune_attr tune = query.is_arc()
        ? sycl_scaleup_tune_attr{ scaleup_protocol::ll, scaleup_algo::ring }
        : sycl_scaleup_tune_attr{ scaleup_protocol::default_kernel, scaleup_algo::unspecified };
    log_tune(query, "reduce", "", tune);
    return tune;
}

}  // namespace ccl::sycl_policy_resolver
