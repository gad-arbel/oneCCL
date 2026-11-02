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

#include <arpa/inet.h>
#include <cstring>
#include <ifaddrs.h>
#include <list>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "occp_kvs.h"
#include "occp/occp/occp_server.h"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace {

const char address_prefix[] = "occp:";
constexpr size_t ip_str_len = 128;
constexpr char scope_id_delim = '%';
constexpr uint32_t server_batch_size = 8;

const std::string CCL_KVS_PREFER_IPV6_ENV = "CCL_KVS_PREFER_IPV6";
const std::string CCL_KVS_IFACE_ENV = "CCL_KVS_IFACE";

[[noreturn]] void throw_unsupported() {
    CCL_THROW("generic kvs get/set is not supported under CCL_KVS_MODE=occp;"
              " set CCL_KVS_MODE=pmi to use ccl::kvs as a key-value store");
}

/* Selects the IP address to advertise as the OCCP server endpoint.
 *
 * Mirrors internal_kvs::fill_local_host_ip: CCL_KVS_IFACE picks an interface by
 * substring, otherwise everything but loopback is eligible, and IPv4 is
 * preferred unless CCL_KVS_PREFER_IPV6 is set. The textual scope of an IPv6
 * address is replaced with the numeric sin6_scope_id so that the published
 * string can be parsed back on another rank.
 */
kvs_status_t select_advertised_ip(std::string& ip) {
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) < 0) {
        LOG_ERROR("can not get host IP");
        return KVS_STATUS_FAILURE;
    }

    const char loopback_name[] = "lo";
    char* iface_name_env = std::getenv(CCL_KVS_IFACE_ENV.c_str());

    std::list<std::string> ipv4s;
    std::list<std::string> ipv6s;
    kvs_status_t status = KVS_STATUS_SUCCESS;

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        bool is_supported_iface = iface_name_env ? strstr(ifa->ifa_name, iface_name_env) != nullptr
                                                 : strstr(ifa->ifa_name, loopback_name) == nullptr;
        if (!is_supported_iface) {
            continue;
        }

        int family = ifa->ifa_addr->sa_family;
        if (family != AF_INET && family != AF_INET6) {
            continue;
        }

        char local_ip[ip_str_len] = { 0 };
        int res = getnameinfo(
            ifa->ifa_addr,
            (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6),
            local_ip,
            ip_str_len,
            nullptr,
            0,
            NI_NUMERICHOST);
        if (res != 0) {
            LOG_ERROR("getnameinfo error > ", gai_strerror(res));
            status = KVS_STATUS_FAILURE;
            break;
        }

        if (family == AF_INET6) {
            char* scope_id_ptr = strchr(local_ip, scope_id_delim);
            if (scope_id_ptr) {
                uint32_t scope_id = ((struct sockaddr_in6*)(ifa->ifa_addr))->sin6_scope_id;
                snprintf(
                    scope_id_ptr + 1, ip_str_len - (scope_id_ptr + 1 - local_ip), "%u", scope_id);
            }
            ipv6s.push_back(local_ip);
        }
        else {
            ipv4s.push_back(local_ip);
        }
    }

    freeifaddrs(ifaddr);
    KVS_CHECK_STATUS(status, "failed to get local host IP");

    size_t is_prefer_ipv6 = 0;
    char* prefer_ipv6_env = std::getenv(CCL_KVS_PREFER_IPV6_ENV.c_str());
    if (prefer_ipv6_env) {
        KVS_CHECK_STATUS(safe_strtol(prefer_ipv6_env, is_prefer_ipv6),
                         "failed to read prefer_ipv6");
    }

    if (is_prefer_ipv6 && ipv6s.empty()) {
        LOG_WARN("ipv6 addresses are not found, fallback to ipv4");
    }

    const std::list<std::string>& addresses =
        (is_prefer_ipv6 && !ipv6s.empty()) ? ipv6s : (!ipv4s.empty() ? ipv4s : ipv6s);
    if (addresses.empty()) {
        LOG_ERROR("can't find interface ", iface_name_env ? iface_name_env : "", " to get host IP");
        return KVS_STATUS_FAILURE;
    }

    ip = addresses.front();
    return KVS_STATUS_SUCCESS;
}

/* sockaddr_t has no notion of an IPv6 scope, so the address is assembled here */
kvs_status_t to_sockaddr(const std::string& ip, in_port_t port, sockaddr_t& addr) {
    std::string host = ip;
    uint32_t scope_id = 0;

    size_t scope_pos = host.find(scope_id_delim);
    if (scope_pos != std::string::npos) {
        KVS_CHECK_STATUS(safe_strtol(host.c_str() + scope_pos + 1, scope_id),
                         "failed to get scope id");
        host.resize(scope_pos);
    }

    sockaddr_storage storage{};
    if (inet_pton(AF_INET, host.c_str(), &((sockaddr_in*)&storage)->sin_addr) == 1) {
        storage.ss_family = AF_INET;
    }
    else if (inet_pton(AF_INET6, host.c_str(), &((sockaddr_in6*)&storage)->sin6_addr) == 1) {
        storage.ss_family = AF_INET6;
        ((sockaddr_in6*)&storage)->sin6_scope_id = scope_id;
    }
    else {
        LOG_ERROR("invalid ip address: ", ip);
        return KVS_STATUS_FAILURE;
    }

    addr = storage;
    addr.port(port);

    return KVS_STATUS_SUCCESS;
}

} // namespace

occp_kvs::occp_kvs() = default;

occp_kvs::~occp_kvs() = default;

bool occp_kvs::is_occp_address(const char* main_addr) {
    return main_addr != nullptr && strncmp(main_addr, address_prefix, strlen(address_prefix)) == 0;
}

kvs_status_t occp_kvs::kvs_main_server_address_reserve(char* main_addr) {
    if (server) {
        LOG_ERROR("occp server is already started at ", published_address);
        return KVS_STATUS_FAILURE;
    }

    std::string ip;
    KVS_CHECK_STATUS(select_advertised_ip(ip), "failed to select an IP to advertise");
    /* port 0: the kernel assigns a free one, which is read back below */
    KVS_CHECK_STATUS(to_sockaddr(ip, 0, server_address), "failed to build the server address");

    const auto& env = ccl::global_data::env();
    server.reset(new occp_server_t(server_address,
                                   static_cast<uint32_t>(env.occp_server_io_threads),
                                   server_batch_size,
                                   static_cast<uint32_t>(env.occp_server_op_timeout),
                                   static_cast<uint32_t>(env.occp_server_ranks_per_thread)));

    in_port_t port = (*server)->local_addr.port();
    if (port == 0) {
        LOG_ERROR("occp server is not bound to a port");
        return KVS_STATUS_FAILURE;
    }
    server_address.port(port);

    published_address = std::string(address_prefix) + ip + ":" + std::to_string(port);
    if (published_address.size() + 1 > ccl::v1::kvs::address_max_size) {
        LOG_ERROR("occp server address ",
                  published_address,
                  " does not fit into ",
                  ccl::v1::kvs::address_max_size,
                  " bytes");
        return KVS_STATUS_FAILURE;
    }

    memset(main_addr, '\0', ccl::v1::kvs::address_max_size);
    kvs_str_copy(main_addr, published_address.c_str(), ccl::v1::kvs::address_max_size);

    LOG_INFO("occp server listens on ", published_address);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t occp_kvs::kvs_init(const char* main_addr) {
    if (main_addr == nullptr) {
        LOG_ERROR("occp server address is not set");
        return KVS_STATUS_FAILURE;
    }

    if (server) {
        /* minting side: the address we published is the one being initialized */
        if (published_address != main_addr) {
            LOG_ERROR(
                "occp server address mismatch: published ", published_address, ", got ", main_addr);
            return KVS_STATUS_FAILURE;
        }
        return KVS_STATUS_SUCCESS;
    }

    std::string addr(main_addr, strnlen(main_addr, ccl::v1::kvs::address_max_size));
    if (!is_occp_address(addr.c_str())) {
        LOG_ERROR("not an occp server address: ", addr);
        return KVS_STATUS_FAILURE;
    }

    /* <prefix><ip>:<port>, the last colon separates the port from an IPv6 address */
    std::string endpoint = addr.substr(strlen(address_prefix));
    size_t port_pos = endpoint.rfind(':');
    if (port_pos == std::string::npos) {
        LOG_ERROR("occp server address must be ", address_prefix, "<ip>:<port>, got ", addr);
        return KVS_STATUS_FAILURE;
    }

    size_t port = 0;
    KVS_CHECK_STATUS(safe_strtol(endpoint.c_str() + port_pos + 1, port),
                     "failed to get the occp server port");
    if (port == 0 || port > UINT16_MAX) {
        LOG_ERROR("unexpected occp server port ", port, " in ", addr);
        return KVS_STATUS_FAILURE;
    }

    KVS_CHECK_STATUS(
        to_sockaddr(endpoint.substr(0, port_pos), static_cast<in_port_t>(port), server_address),
        "failed to parse the occp server address");

    published_address = addr;
    LOG_INFO("occp server address ", published_address);

    return KVS_STATUS_SUCCESS;
}

kvs_status_t occp_kvs::kvs_finalize() {
    /* the server has to outlive the bootstrap of every rank, so it is only
     * released together with this object */
    return KVS_STATUS_SUCCESS;
}

kvs_status_t occp_kvs::kvs_set_value(const std::string&, const std::string&, const std::string&) {
    throw_unsupported();
}

kvs_status_t occp_kvs::kvs_remove_name_key(const std::string&, const std::string&) {
    throw_unsupported();
}

kvs_status_t occp_kvs::kvs_get_value_by_name_key(const std::string&,
                                                 const std::string&,
                                                 std::string&) {
    throw_unsupported();
}

kvs_status_t occp_kvs::kvs_get_count_names(const std::string&, size_t&) {
    throw_unsupported();
}

kvs_status_t occp_kvs::kvs_get_keys_values_by_name(const std::string&,
                                                   std::vector<std::string>&,
                                                   std::vector<std::string>&,
                                                   size_t&) {
    throw_unsupported();
}

kvs_status_t occp_kvs::kvs_get_replica_size(size_t&) {
    throw_unsupported();
}
