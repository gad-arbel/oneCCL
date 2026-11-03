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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "comm/comm.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_policy.hpp"  // ccl::sycl_policy::scaleout_algo
#include "coll/selection/selection.hpp"

// dispatching between SYCL path and Scheduler path
bool can_use_sycl_kernels(const ccl_selector_param& param);

using sycl_scaleout_tune_attr = ccl::sycl_policy::sycl_scaleout_tune_attr;


class sycl_scaleout_selector {
public:
    sycl_scaleout_selector(size_t size,
                           size_t comm_size,
                           ccl_datatype dtype,
                           bool is_recording) noexcept
        : size(size), comm_size(comm_size), dtype(dtype), is_recording(is_recording) {}

    sycl_scaleout_tune_attr allreduce()      const;
    sycl_scaleout_tune_attr reduce_scatter() const;
    sycl_scaleout_tune_attr allgatherv()     const;
    sycl_scaleout_tune_attr alltoall()       const;
    sycl_scaleout_tune_attr scatter()        const;
    sycl_scaleout_tune_attr broadcast()      const;
    sycl_scaleout_tune_attr reduce()         const;

    static size_t allreduce_chunk_size(ccl::sycl_policy::scaleout_algo algo,
                                       size_t size,
                                       size_t comm_size);
    static size_t reduce_scatter_chunk_size(ccl::sycl_policy::scaleout_algo algo,
                                            size_t size,
                                            size_t comm_size);
    static size_t allgatherv_chunk_size(ccl::sycl_policy::scaleout_algo algo,
                                        size_t size,
                                        size_t comm_size);

    size_t size;
    size_t comm_size;
    ccl_datatype dtype;
    bool is_recording;
};

// Per-message-size chunk sizes for the CCL_SYCL_<COLL>_SIMPLE_CHUNK_SIZE envs,
// parsed with the same "<value>:<lo>-<hi>;..." syntax as CCL_SYCL_ALLREDUCE_LL.
// Values are chunk sizes in bytes; 0 (or an empty spec) disables chunking.
using simple_chunk_size_table_t = ccl_selection_table_t<size_t>;
simple_chunk_size_table_t parse_simple_chunk_size_spec(const char* env_name,
                                                       const std::string& spec);
size_t select_simple_chunk_size(const simple_chunk_size_table_t& table, size_t msg_bytes);
