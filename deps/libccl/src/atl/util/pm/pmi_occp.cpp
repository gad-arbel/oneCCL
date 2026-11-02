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

#include "atl/util/pm/pmi_occp.h"
#include "common/global/global.hpp"

namespace {
constexpr uint32_t client_batch_size = 2;
}

pmi_occp::pmi_occp(int comm_size, const std::vector<int>& ranks, std::shared_ptr<occp_kvs> k)
        : size(comm_size),
          k(std::move(k)) {
    CCL_THROW_IF_NOT(ranks.size() == 1,
                     "occp bootstrap supports one rank per communicator, got ",
                     ranks.size(),
                     "; set CCL_KVS_MODE=pmi for several ranks per process");
    CCL_THROW_IF_NOT(this->k != nullptr, "occp kvs is not set");

    rank = ranks.front();
}

pmi_occp::~pmi_occp() {
    if (!is_finalized) {
        pmrt_finalize();
    }
}

atl_status_t pmi_occp::pmrt_init() {
    const auto& env = ccl::global_data::env();

    client.reset(new occp_client_t(rank,
                                   k->get_server_address(),
                                   static_cast<uint32_t>(env.occp_client_io_threads),
                                   client_batch_size,
                                   static_cast<uint32_t>(env.occp_client_op_timeout)));

    LOG_DEBUG(
        "occp client: rank ", rank, ", size ", size, ", server ", k->get_server_address().str());

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_occp::pmrt_finalize() {
    is_finalized = true;
    client.reset();

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_occp::pmrt_barrier() {
    CCL_THROW_IF_NOT(client != nullptr, "occp client is not initialized");

    return client->rendezvous(rank, size) ? ATL_STATUS_SUCCESS : ATL_STATUS_FAILURE;
}

int pmi_occp::get_rank() {
    return rank;
}

int pmi_occp::get_size() {
    return size;
}

atl_status_t pmi_occp::pmrt_kvs_put(char*, int, const void*, size_t) {
    LOG_ERROR("unsupported: occp bootstrap does not provide a key-value store");
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_occp::pmrt_kvs_get(char*, int, void*, size_t) {
    LOG_ERROR("unsupported: occp bootstrap does not provide a key-value store");
    return ATL_STATUS_UNSUPPORTED;
}

int pmi_occp::is_pm_resize_enabled() {
    return 0;
}

atl_status_t pmi_occp::pmrt_main_addr_reserve(char*) {
    LOG_ERROR("unsupported");
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_occp::pmrt_set_resize_function(atl_resize_fn_t) {
    LOG_ERROR("unsupported");
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_occp::pmrt_update() {
    LOG_ERROR("unsupported");
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_occp::pmrt_wait_notification() {
    LOG_ERROR("unsupported");
    return ATL_STATUS_UNSUPPORTED;
}

size_t pmi_occp::get_local_thread_idx() {
    return 0;
}

atl_status_t pmi_occp::get_local_kvs_id(size_t& res) {
    res = 0;
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_occp::set_local_kvs_id(size_t) {
    return ATL_STATUS_SUCCESS;
}

size_t pmi_occp::get_threads_per_process() {
    return 1;
}

size_t pmi_occp::get_ranks_per_process() {
    return 1;
}
