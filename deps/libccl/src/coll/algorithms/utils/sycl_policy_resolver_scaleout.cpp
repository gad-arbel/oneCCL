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

#include "coll/algorithms/utils/sycl_policy_resolver_scaleout.hpp"

#include "coll/algorithms/utils/sycl_policy.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_json.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "proxy/proxy.hpp"

#include <algorithm>
#include <initializer_list>
#include <optional>

namespace ccl::sycl_policy_resolver {

using sycl_policy::scaleout_algo;

namespace {

// Logs the resolver's decision.
// `source` is "" for the C++ selectors and "[json]" for a policy decision.
void log_tune(const sycl_policy::policy_query& query,
              const char* coll,
              const char* source,
              const sycl_scaleout_tune_attr& tune) {
    LOG_DEBUG("scaleout ", coll, source,
              " algo=",         sycl_policy::to_str(tune.algo()),
              " chunk_bytes=",  tune.pipeline_chunk_size(),
              " msg_bytes=",    query.msg_bytes,
              " r2r_size=",     query.r2r_comm_size,
              " global_size=",  query.global_comm_size,
              " is_recording=", query.is_recording);
}

// alltoall(v) algorithms that require GPU RDMA.
bool needs_gpu_rdma(scaleout_algo a) {
    return a == scaleout_algo::scatter || a == scaleout_algo::gdr_only_pairwise ||
           a == scaleout_algo::numa_gdr_split;
}

// The dispatchers run the exchange on the proxy thread only when it is
// requested and the service started.
bool uses_proxy(bool requested) {
    const auto& proxy = ccl::global_data::get().proxy_service;
    return requested && proxy && proxy->is_started();
}

// Mirrors the C++ selectors under SYCL-graph capture: allreduce, reduce_scatter,
// allgatherv and alltoall(v) run only `direct` or an algorithm whose exchange
// runs on the proxy thread (CCL_SYCL_ENABLE_<COLL>_SCALEOUT_PROXY). scatter,
// broadcast and reduce have no recording restriction.
bool runs_under_recording(ccl_coll_type ctype, scaleout_algo a) {
    if (a == scaleout_algo::direct) return true;
    const auto& env = ccl::global_data::env();
    switch (ctype) {
        case ccl_coll_allreduce:
            return (a == scaleout_algo::ring || a == scaleout_algo::rabenseifner) &&
                   uses_proxy(env.sycl_enable_allreduce_scaleout_proxy);
        case ccl_coll_reduce_scatter:
            return a == scaleout_algo::ring &&
                   uses_proxy(env.sycl_enable_reduce_scatter_scaleout_proxy);
        case ccl_coll_allgatherv:
            return a == scaleout_algo::ring &&
                   uses_proxy(env.sycl_enable_allgatherv_scaleout_proxy);
        case ccl_coll_alltoall:
        case ccl_coll_alltoallv:
            return a == scaleout_algo::scatter &&
                   uses_proxy(env.sycl_enable_alltoall_scaleout_proxy);
        default: return true;
    }
}

// Returns the JSON decision if `coll` supports it and it can run, otherwise nullopt.
std::optional<sycl_scaleout_tune_attr> from_json(const sycl_policy::policy_query& query,
                                                 const char* coll,
                                                 std::initializer_list<scaleout_algo> supported) {
    auto tune = json::scaleout(query);
    if (!tune) return tune;
    if (std::find(supported.begin(), supported.end(), tune->algo()) == supported.end()) {
        LOG_WARN_ONCE("scaleout ", coll, "[json] algo '", sycl_policy::to_str(tune->algo()),
                      "' not supported, falling back; ");
        return std::nullopt;
    }
    if (needs_gpu_rdma(tune->algo()) && !ccl::global_data::env().sycl_enable_direct_gpu_rdma) {
        LOG_WARN_ONCE("scaleout ", coll, "[json] algo '", sycl_policy::to_str(tune->algo()),
                      "' needs CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1, falling back; ");
        return std::nullopt;
    }
    if (query.is_recording && !runs_under_recording(query.ctype, tune->algo())) {
        LOG_WARN_ONCE("scaleout ", coll, "[json] algo '", sycl_policy::to_str(tune->algo()),
                      "' cannot run under SYCL-graph capture without its "
                      "scaleout proxy, falling back; ");
        return std::nullopt;
    }
    log_tune(query, coll, "[json]", *tune);
    return tune;
}

}  // namespace

sycl_scaleout_tune_attr scaleout_resolver::allgatherv() const {
    if (auto json_hit = from_json(query, "allgatherv",
                                  { scaleout_algo::direct, scaleout_algo::ring })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.allgatherv();
    log_tune(query, "allgatherv", "", tune);
    return tune;
}

sycl_scaleout_tune_attr scaleout_resolver::allreduce() const {
    if (auto json_hit = from_json(query, "allreduce",
                                  { scaleout_algo::direct, scaleout_algo::ring,
                                    scaleout_algo::rabenseifner })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.allreduce();
    log_tune(query, "allreduce", "", tune);
    return tune;
}

sycl_scaleout_tune_attr scaleout_resolver::reduce_scatter() const {
    if (auto json_hit = from_json(query, "reduce_scatter",
                                  { scaleout_algo::direct, scaleout_algo::ring })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.reduce_scatter();
    log_tune(query, "reduce_scatter", "", tune);
    return tune;
}

sycl_scaleout_tune_attr scaleout_resolver::alltoall() const {
    if (auto json_hit = from_json(query, "alltoall",
                                  { scaleout_algo::direct, scaleout_algo::scatter,
                                    scaleout_algo::gdr_only_pairwise,
                                    scaleout_algo::numa_gdr_split, scaleout_algo::non_sycl })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.alltoall();
    log_tune(query, "alltoall", "", tune);
    return tune;
}

// Same as alltoall, without numa_gdr_split.
sycl_scaleout_tune_attr scaleout_resolver::alltoallv() const {
    if (auto json_hit = from_json(query, "alltoallv",
                                  { scaleout_algo::direct, scaleout_algo::scatter,
                                    scaleout_algo::gdr_only_pairwise, scaleout_algo::non_sycl })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.alltoall();
    log_tune(query, "alltoallv", "", tune);
    return tune;
}

// Multi-node scatter: `direct` or `scatter_two_phase`.
sycl_scaleout_tune_attr scaleout_resolver::scatter() const {
    if (auto json_hit = from_json(query, "scatter",
                                  { scaleout_algo::direct, scaleout_algo::scatter_two_phase })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.scatter();
    log_tune(query, "scatter", "", tune);
    return tune;
}

// Multi-node broadcast: `direct` or `scatter_allgather` (Arc only).
sycl_scaleout_tune_attr scaleout_resolver::broadcast() const {
    if (auto json_hit = from_json(query, "broadcast",
                                  { scaleout_algo::direct, scaleout_algo::scatter_allgather })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.broadcast();
    log_tune(query, "broadcast", "", tune);
    return tune;
}

// Multi-node reduce has only `direct`.
sycl_scaleout_tune_attr scaleout_resolver::reduce() const {
    if (auto json_hit = from_json(query, "reduce", { scaleout_algo::direct })) {
        return *json_hit;
    }
    auto tune = sycl_scaleout_selector{query.msg_bytes, query.r2r_comm_size,
                                       query.dtype,     query.is_recording}.reduce();
    log_tune(query, "reduce", "", tune);
    return tune;
}

}  // namespace ccl::sycl_policy_resolver
