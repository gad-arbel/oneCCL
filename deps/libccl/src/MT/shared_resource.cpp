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

#include "MT/shared_resource.hpp"
#include "common/log/log.hpp"
#include "shared_resource.hpp"

namespace ccl {

shared_resources::~shared_resources() {
    hash_table.clear();
    pt2pt_rendezvous.clear();
    barrier_waits.clear();
    rank_info_vec_globs.clear();
    barrier_initialized_flags.clear();
    resource_mutexes.clear();
    all_hostnames_raw_globs.clear();
}

pt2pt_rendezvous_entry &shared_resources::get_pt2pt_rendezvous(int global_id,
                                                               int src,
                                                               int dst) {
    CCL_THROW_IF_NOT(global_id >= 0, "get_pt2pt_rendezvous: invalid global_id ", global_id);
    CCL_THROW_IF_NOT(src >= 0 && src <= 0xffff && dst >= 0 && dst <= 0xffff,
                     "get_pt2pt_rendezvous: src/dst out of range src=",
                     src,
                     " dst=",
                     dst);
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(global_id)) << 32) |
                         (static_cast<uint64_t>(static_cast<uint16_t>(src)) << 16) |
                         static_cast<uint64_t>(static_cast<uint16_t>(dst));
    std::lock_guard<std::mutex> lk(pt2pt_rendezvous_mutex);
    return pt2pt_rendezvous[key];
}

void shared_resources::resize_rank_info_vec_glob(size_t new_size, int global_id) {
    std::lock_guard<std::mutex> lock(rank_info_mutex);
    rank_info_vec_globs[global_id].resize(new_size);
}

void shared_resources::resize_all_hostnames_raw(size_t new_size, int global_id) {
    std::lock_guard<std::mutex> lock(hostnames_mutex);
    all_hostnames_raw_globs[global_id].resize(new_size);
}

std::mutex &shared_resources::get_resource_mutex(int global_id) {
    static std::mutex mu;
    std::lock_guard<std::mutex> guard(mu);
    return resource_mutexes[global_id];
}

std::vector<char> &shared_resources::get_all_hostnames_raw_glob(int global_id) {
    std::lock_guard<std::mutex> lock(hostnames_mutex);
    return all_hostnames_raw_globs[global_id];
}

rank_info_vec_t &shared_resources::get_rank_info_vec_glob(int global_id) {
    std::lock_guard<std::mutex> lock(rank_info_mutex);
    return rank_info_vec_globs[global_id];
}

void shared_resources::init_barrier_wait(int threads_num, int global_id) {
    pthread_barrier_t *barrier_ptr = nullptr;
    {
        std::lock_guard<std::mutex> guard(barrier_waits_mutex);
        if (barrier_initialized_flags.find(global_id) == barrier_initialized_flags.end() ||
            !barrier_initialized_flags[global_id]) {
            barrier_waits[global_id] = std::make_unique<pthread_barrier_t>();
            int rc = pthread_barrier_init(barrier_waits[global_id].get(), nullptr, threads_num);
            CCL_THROW_IF_NOT(rc == 0, "pthread_barrier_init failed, rc: ", rc);
            barrier_initialized_flags[global_id] = true;
        }
        barrier_ptr = barrier_waits[global_id].get();
    }

    pthread_barrier_wait(barrier_ptr);
}

pthread_barrier_t *shared_resources::get_barrier(int global_id) {
    std::lock_guard<std::mutex> guard(barrier_waits_mutex);
    auto it = barrier_waits.find(global_id);
    CCL_THROW_IF_NOT(it != barrier_waits.end() && it->second,
                     "barrier_waits is not initialized for global_id ",
                     global_id);
    return it->second.get();
}

int shared_resources::get_node_rank(int ranks[2], int pair_comm_size) {
    // Possibly this calculation has to be universalized
    return ranks[0] * pair_comm_size + ranks[1];
}

} // namespace ccl
