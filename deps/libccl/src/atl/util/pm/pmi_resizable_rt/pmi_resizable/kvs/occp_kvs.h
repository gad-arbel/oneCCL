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
#include <string>

#include "ikvs_wrapper.h"
#include "sockaddr.h"

class occp_server_t;

/* kvs that carries the address of a per-communicator OCCP server.
 *
 * Minting a unique id (ccl::create_main_kvs, i.e. onecclGetUniqueId) starts an
 * OCCP server on an ephemeral port and publishes "occp:<ip>:<port>" into the
 * address blob. A rank that attaches to that blob only parses the address out
 * of it: the bootstrap itself runs over OCCP (see pmi_occp), so the generic
 * key-value operations of ikvs_wrapper are not available here.
 */
class occp_kvs final : public ikvs_wrapper {
public:
    /* defined out of line: occp_server_t is only complete in the source file */
    occp_kvs();

    ~occp_kvs() override;

    occp_kvs(const occp_kvs&) = delete;

    occp_kvs& operator=(const occp_kvs&) = delete;

    /* starts the server and writes its address into main_addr */
    kvs_status_t kvs_main_server_address_reserve(char* main_addr) override;

    kvs_status_t kvs_init(const char* main_addr) override;

    kvs_status_t kvs_finalize() override;

    kvs_status_t kvs_set_value(const std::string& kvs_name,
                               const std::string& kvs_key,
                               const std::string& kvs_val) override;

    kvs_status_t kvs_remove_name_key(const std::string& kvs_name,
                                     const std::string& kvs_key) override;

    kvs_status_t kvs_get_value_by_name_key(const std::string& kvs_name,
                                           const std::string& kvs_key,
                                           std::string& kvs_val) override;

    kvs_status_t kvs_get_count_names(const std::string& kvs_name, size_t& count_names) override;

    kvs_status_t kvs_get_keys_values_by_name(const std::string& kvs_name,
                                             std::vector<std::string>& kvs_keys,
                                             std::vector<std::string>& kvs_values,
                                             size_t& count) override;

    kvs_status_t kvs_get_replica_size(size_t& replica_size) override;

    static bool is_occp_address(const char* main_addr);

    const sockaddr_t& get_server_address() const {
        return server_address;
    }

private:
    std::unique_ptr<occp_server_t> server;
    sockaddr_t server_address;
    std::string published_address;
};
