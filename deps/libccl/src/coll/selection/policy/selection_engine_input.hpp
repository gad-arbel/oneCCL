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

// Input struct for SelectionEngine::resolve().

#include <cstddef>
#include <optional>

#include "coll/algorithms/algorithm_utils.hpp"  // ccl_coll_type
#include "common/datatype/datatype.hpp"         // ccl_datatype

namespace ccl {
namespace selection {

// Minimal input struct the engine reads. Callers fill only the axes the JSON
// filters actually match against.
struct selection_engine_input {
    enum class phase_t         : int { scaleup = 0, scaleout = 1 };
    // Values mirror ccl_p2p_domain_level so callers can static_cast when known.
    enum class p2p_domain_t    : int { p2p_auto = 0, p2p_node, p2p_numa, p2p_switch, p2p_none };
    // Values mirror ccl_atl_transport.
    enum class transport_t     : int { ofi = 0, mpi = 1 };

    ccl_coll_type    ctype     = ccl_coll_last_value;
    std::size_t      msg_bytes = 0;   // per-rank bytes (count * dtype_size)
    // Per-rank send-buffer bytes; set only for scaleup queries (reduce_scatter:
    // recv_count * ranks). Unset never matches.
    std::optional<std::size_t> send_msg_bytes;
    std::size_t      count     = 0;
    std::size_t      comm_size = 0;
    ccl_datatype     dtype     = ccl_datatype_int8;
    std::optional<phase_t>         phase;
    std::optional<p2p_domain_t>    p2p_domain;
    std::optional<bool>            has_p2p_access;
    std::optional<bool>            is_multi_thread;
    std::optional<bool>            is_arc;
    std::optional<bool>            is_recording;
    std::optional<transport_t>     transport;
    std::optional<bool>            direct_gpu_rdma;
    std::optional<bool>            use_copy_engine;
};

}  // namespace selection
}  // namespace ccl
