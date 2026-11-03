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

#include <cstddef>
#include <cstdint>

#include "coll/algorithms/algorithm_utils.hpp"    // ccl_coll_type
#include "common/datatype/datatype.hpp"           // ccl_datatype
#include "comm/comm.hpp"                          // ccl_comm topology accessors
#include "common/stream/stream.hpp"               // ccl_stream::get_ze_device
#include "common/global/ze/ze_data.hpp"           // ccl::ze::get_device_family
#include "sched/entry/ze/ze_primitives.hpp"       // is_arc_card
#include "coll/selection/policy/selection_engine_input.hpp"   // selection::selection_engine_input
#include "common/env/env.hpp"                     // ccl_selection_table_t
#include "coll/algorithms/utils/sycl_pipeline_chunk.hpp"  // default_pipeline_chunk_size
#include "common/global/global.hpp"               // ccl::global_data::env
#include "coll/algorithms/utils/p2p_domain.hpp"     // ccl_p2p_domain_level

#include <optional>
#include <string>

class ccl_comm;
class ccl_stream;

namespace ccl::sycl_policy {

// -----------------------------------------------------------------------------
// Scaleout vocabulary (used by the scaleout auto-tuner and resolver).
// -----------------------------------------------------------------------------

enum class scaleout_algo : std::uint8_t {
    ring,                  // pipelined ring over r2r, chunked by pipeline_chunk_size
    direct,                // one ATL collective call; host-staged unless GPU RDMA is on
    rabenseifner,          // allreduce: recursive halving/doubling (RS + AG), pipelined
    scatter_two_phase,     // scatter multi-node: node scatter, then chunked r2r scatter
    gdr_only_pairwise,     // alltoall: per-peer send/recv steps, all scaleout; GPU RDMA
    scatter,               // alltoall: post all sends/recvs at once, all scaleout; GPU RDMA
    numa_gdr_split,        // alltoall: IPC within NUMA, GPU RDMA to the rest
    non_sycl,              // decline the SYCL path; run the scheduler implementation
    scatter_allgather,     // broadcast multi-node: local scatter, rail bcast, local allgather
};

// -----------------------------------------------------------------------------
// Scaleup vocabulary — two orthogonal axes returned by scaleup_resolver.
// -----------------------------------------------------------------------------

// Family of scaleup kernels the caller should dispatch into.
enum class scaleup_protocol : std::uint8_t {
    ll,              // Arc LL256 family
    default_kernel,  // non-Arc SYCL small/large kernels
    copy_engine,     // copy engine queues (CCL_ENABLE_ZE_CE), Arc only
};

enum class scaleup_algo : std::uint8_t {
    unspecified,     // resolver doesn't select an algo — dispatcher decides
    // ll protocol sub-choices:
    ring,
    oneshot,
    twoshots,
    // default protocol sub-choices:
    small,
    large,
    // copy_engine protocol sub-choices:
    sycl_copy_engine,
    l0_copy_engine,
};

constexpr bool is_size_algo(scaleup_algo a) noexcept {
    return a == scaleup_algo::small || a == scaleup_algo::large;
}

constexpr bool is_copy_engine_algo(scaleup_algo a) noexcept {
    return a == scaleup_algo::sycl_copy_engine || a == scaleup_algo::l0_copy_engine;
}

// Names match the JSON policy vocabulary.
const char* to_str(scaleout_algo a) noexcept;
const char* to_str(scaleup_protocol p) noexcept;
const char* to_str(scaleup_algo a) noexcept;

// Bucket `msg_bytes` into a size algo.
constexpr scaleup_algo pick_size_algo(std::size_t msg_bytes,
                                      std::size_t small_threshold) noexcept {
    return msg_bytes <= small_threshold ? scaleup_algo::small : scaleup_algo::large;
}


// Per-chunk LL algorithm lookup; defaults to `ring`. `fixed()` pins one
// algorithm for all chunks.
class ll_algo_lookup {
public:
    ll_algo_lookup() = default;
    explicit ll_algo_lookup(const ccl_selection_table_t<std::string>& t) noexcept
        : table(&t) {}
    // Holds a pointer: a temporary table would dangle.
    explicit ll_algo_lookup(ccl_selection_table_t<std::string>&&) = delete;

    static ll_algo_lookup fixed(scaleup_algo a) noexcept {
        ll_algo_lookup l;
        l.fixed_algo = a;
        return l;
    }

    scaleup_algo pick(std::size_t chunk_bytes) const;

    const ccl_selection_table_t<std::string>* table = nullptr;
    scaleup_algo fixed_algo = scaleup_algo::unspecified;
};

class sycl_scaleup_tune_attr {
public:
    sycl_scaleup_tune_attr(scaleup_protocol p, scaleup_algo a,
                            ll_algo_lookup l = {}) noexcept
        : protocol(p), algo(a), ll(l) {}

    scaleup_protocol protocol = scaleup_protocol::default_kernel;
    scaleup_algo     algo     = scaleup_algo::unspecified;
    ll_algo_lookup   ll;
};

//   allreduce:      algo (direct|rabenseifner|ring), chunk
//   reduce_scatter: algo (direct|ring),              chunk
//   allgatherv:     algo (direct|ring),              chunk
//   alltoall(v):    algo (direct|gdr_only_pairwise|scatter|numa_gdr_split|non_sycl)
//   scatter:        algo (direct|scatter_two_phase)
//   broadcast:      algo (direct|scatter_allgather)
//   reduce:         algo (direct)

class sycl_scaleout_tune_attr {
public:
    sycl_scaleout_tune_attr() = default;
    explicit sycl_scaleout_tune_attr(scaleout_algo a) noexcept
        : selected_algo(a) {}
    sycl_scaleout_tune_attr(scaleout_algo a, size_t chunk) noexcept
        : selected_algo(a), chunk_bytes(chunk) {}

    scaleout_algo algo()                const noexcept { return selected_algo; }
    size_t    pipeline_chunk_size() const noexcept { return chunk_bytes; }

    sycl_scaleout_tune_attr& set_algo(scaleout_algo a) noexcept             { selected_algo = a; return *this; }
    sycl_scaleout_tune_attr& set_pipeline_chunk_size(size_t s) noexcept { chunk_bytes = s; return *this; }

private:
    scaleout_algo selected_algo{ scaleout_algo::direct };
    size_t        chunk_bytes{ default_pipeline_chunk_size };
};


class policy_query {
public:
    policy_query(ccl_coll_type ct,
                 std::size_t   element_count,
                 ccl_datatype  data_type,
                 std::size_t   r2r_size,
                 std::size_t   global_size,
                 bool          rec = false)
        : ctype(ct),
          msg_bytes(element_count * data_type.size()),
          count(element_count),
          dtype(data_type),
          r2r_comm_size(r2r_size == 0 ? global_size : r2r_size),
          global_comm_size(global_size),
          is_recording(rec) {}

    // Opt-in setters for topology / stream / comm.
    policy_query& with_comm(ccl_comm* c) noexcept       { comm = c;          return *this; }
    // `p2p_access_from_all`: has_p2p_access() checks all ranks of `comm`, not just `p2p_comm`.
    policy_query& with_p2p_comm(ccl_comm* p, bool p2p_access_from_all = false) noexcept {
        p2p_comm = p;
        this->p2p_access_from_all = p2p_access_from_all;
        return *this;
    }
    policy_query& with_stream(ccl_stream* s) noexcept   { global_stream = s; return *this; }

    // For queries without `p2p_comm`.
    policy_query& with_p2p_level(std::optional<ccl_p2p_domain_level> l) noexcept {
        p2p_level = l;
        return *this;
    }

    // Per-rank send-buffer bytes; reduce_scatter sends recv_count from every p2p rank.
    std::size_t scaleup_send_bytes() const noexcept {
        if (ctype == ccl_coll_reduce_scatter && p2p_comm) {
            return msg_bytes * static_cast<std::size_t>(p2p_comm->size());
        }
        return msg_bytes;
    }

    bool is_arc() const {
        if (!global_stream) return false;
        if (!is_arc_cache) {
            is_arc_cache =
                ccl::ze::is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));
        }
        return *is_arc_cache;
    }

    bool is_multi_thread() const {
        return comm && comm->is_multi_thread_instance();
    }

    // From `p2p_comm` when set (unless p2p_access_from_all), else `comm`.
    bool has_p2p_access() const {
        ccl_comm* c = (p2p_comm && !p2p_access_from_all) ? p2p_comm : comm;
        return c && c->get_topo_manager().has_p2p_access();
    }

    // Build the JSON-engine input from this query. `phase` picks which comm-size
    // axis the engine sees (scaleup → p2p comm; scaleout → r2r comm). Optional
    // fields are populated only when their underlying source is wired — an
    // unwired axis stays nullopt so JSON filters like `is_arc=yes` skip cleanly.
    ccl::selection::selection_engine_input to_selection_engine_input(
        ccl::selection::selection_engine_input::phase_t phase) const {
        const bool scaleup = phase == ccl::selection::selection_engine_input::phase_t::scaleup;
        ccl::selection::selection_engine_input in;
        in.ctype     = ctype;
        in.msg_bytes = msg_bytes;
        in.count     = count;
        if (scaleup) in.send_msg_bytes = scaleup_send_bytes();
        in.dtype     = dtype;
        in.comm_size = scaleup ? (p2p_comm ? static_cast<std::size_t>(p2p_comm->size())
                                           : global_comm_size)
                               : r2r_comm_size;
        in.phase        = phase;
        in.is_recording = is_recording;
        if (global_stream)      in.is_arc          = is_arc();
        if (comm)               in.is_multi_thread = is_multi_thread();
        if (comm || p2p_comm)   in.has_p2p_access  = has_p2p_access();
        const auto& env    = ccl::global_data::env();
        in.transport       = static_cast<ccl::selection::selection_engine_input::transport_t>(
            env.atl_transport);
        in.direct_gpu_rdma = env.sycl_enable_direct_gpu_rdma;
        in.use_copy_engine = env.enable_ze_ce;
        const auto level = p2p_level ? p2p_level
                           : p2p_comm ? p2p_comm->get_p2p_level()
                                      : std::nullopt;
        if (level) {
            in.p2p_domain =
                static_cast<ccl::selection::selection_engine_input::p2p_domain_t>(*level);
        }
        return in;
    }

    ccl_coll_type ctype;
    std::size_t   msg_bytes;        // count * dtype size
    std::size_t   count;            // elements
    ccl_datatype  dtype;
    std::size_t   r2r_comm_size;
    std::size_t   global_comm_size;
    bool          is_recording;
    ccl_comm*     comm          = nullptr;
    ccl_comm*     p2p_comm      = nullptr;
    ccl_stream*   global_stream = nullptr;
    std::optional<ccl_p2p_domain_level> p2p_level;

private:
    bool p2p_access_from_all = false;
    mutable std::optional<bool> is_arc_cache;
};

}  // namespace ccl::sycl_policy
