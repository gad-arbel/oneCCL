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

#include <memory>
#include <vector>

#include "atl/util/pm/pm_rt.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/occp_kvs.h"
#include "occp/occp/occp_client.h"

/* PMI backed by OCCP.
 *
 * rank and size are the values the application passed to commInitRank (i.e.
 * ccl::create_communicator), so no process manager is consulted. The exchanges
 * the OFI transport needs - hostnames, endpoint names and a barrier - are served
 * by the OCCP client owned here, which makes the client per-communicator.
 */
class pmi_occp final : public ipmi {
public:
    pmi_occp() = delete;

    pmi_occp(int comm_size, const std::vector<int>& ranks, std::shared_ptr<occp_kvs> k);

    ~pmi_occp() override;

    pmi_occp(const pmi_occp&) = delete;

    pmi_occp& operator=(const pmi_occp&) = delete;

    int is_pm_resize_enabled() override;

    atl_status_t pmrt_init() override;

    atl_status_t pmrt_main_addr_reserve(char* main_addr) override;

    atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) override;

    atl_status_t pmrt_update() override;

    atl_status_t pmrt_wait_notification() override;

    atl_status_t pmrt_barrier() override;

    atl_status_t pmrt_kvs_put(char* kvs_key,
                              int proc_idx,
                              const void* kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_kvs_get(char* kvs_key,
                              int proc_idx,
                              void* kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_finalize() override;

    int get_rank() override;

    int get_size() override;

    size_t get_local_thread_idx() override;

    atl_status_t get_local_kvs_id(size_t& res) override;

    atl_status_t set_local_kvs_id(size_t local_kvs_id) override;

    size_t get_threads_per_process() override;

    size_t get_ranks_per_process() override;

    occp_client_t* get_client() const {
        return client.get();
    }

private:
    bool is_finalized{ false };

    int rank{ 0 };
    int size{ 0 };

    std::shared_ptr<occp_kvs> k;
    std::unique_ptr<occp_client_t> client;
};
