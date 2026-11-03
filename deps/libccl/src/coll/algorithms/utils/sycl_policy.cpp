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

#include "coll/algorithms/utils/sycl_policy.hpp"

#include "coll/selection/selector.hpp"  // ccl_algorithm_selector_base

namespace ccl::sycl_policy {

const char* to_str(scaleout_algo a) noexcept {
    switch (a) {
        case scaleout_algo::ring:              return "ring";
        case scaleout_algo::direct:            return "direct";
        case scaleout_algo::rabenseifner:      return "rabenseifner";
        case scaleout_algo::scatter_two_phase: return "scatter_two_phase";
        case scaleout_algo::gdr_only_pairwise: return "gdr_only_pairwise";
        case scaleout_algo::scatter:           return "scatter";
        case scaleout_algo::numa_gdr_split:    return "numa_gdr_split";
        case scaleout_algo::non_sycl:          return "non_sycl";
        case scaleout_algo::scatter_allgather: return "scatter_allgather";
    }
    return "?";
}

const char* to_str(scaleup_protocol p) noexcept {
    switch (p) {
        case scaleup_protocol::ll:             return "sycl_ll";
        case scaleup_protocol::default_kernel: return "sycl_simple";
        case scaleup_protocol::copy_engine:    return "copy_engine";
    }
    return "?";
}

const char* to_str(scaleup_algo a) noexcept {
    switch (a) {
        case scaleup_algo::unspecified: return "unspecified";
        case scaleup_algo::ring:        return "ring";
        case scaleup_algo::oneshot:     return "oneshot";
        case scaleup_algo::twoshots:    return "twoshots";
        case scaleup_algo::small:       return "small";
        case scaleup_algo::large:       return "large";
        case scaleup_algo::sycl_copy_engine: return "sycl_copy_engine";
        case scaleup_algo::l0_copy_engine:   return "l0_copy_engine";
    }
    return "?";
}

scaleup_algo ll_algo_lookup::pick(std::size_t chunk_bytes) const {
    if (fixed_algo != scaleup_algo::unspecified) return fixed_algo;
    if (!table || table->empty()) return scaleup_algo::ring;
    const std::string& s =
        ccl_algorithm_selector_base<std::string>::get_value_from_table(chunk_bytes, *table);
    if (s == "oneshot")  return scaleup_algo::oneshot;
    if (s == "twoshots") return scaleup_algo::twoshots;
    if (s == "ring")     return scaleup_algo::ring;
    return scaleup_algo::unspecified;
}

}  // namespace ccl::sycl_policy
