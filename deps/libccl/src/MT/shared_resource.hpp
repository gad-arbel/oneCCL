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

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#include "atl/atl_base_comm.hpp"
#include "topology/topo_manager.hpp"

namespace ccl {

// === Handshake data for non-group pt2pt operations ===
// One entry per (global_id, src, dst) triple, so concurrent pt2pt pairs on the
// same communicator (e.g. 0->1 and 2->3) rendezvous independently. Like the
// group path, a single entry supports one outstanding operation per direction
// per pair: the flags are reset by their consumer so the entry can be reused by
// the next send/recv on that pair.
struct pt2pt_rendezvous_entry {
    std::mutex m;
    std::condition_variable cv;

    // Indicate that the receiver has published its buffer pointer
    bool recv_published = false;

    // Indicate that the sender has answered the receiver: either submitted its
    // memcpy, or skipped it because send_bytes != recv_bytes
    bool copy_done = false;

    // Size of the sender's buffer in bytes, published with copy_done. Both sides
    // compare it against recv_bytes, so a size mismatch fails on both ranks
    // instead of leaving the receiver waiting.
    size_t send_bytes = 0;

    // Receiver's buffer, published under the mutex. All MT ranks live in the same
    // process, so the sender can copy into it directly - same as the group path.
    void *recv_ptr = nullptr;
    // Size of the receiver's buffer in bytes, published with recv_ptr
    size_t recv_bytes = 0;
#ifdef CCL_ENABLE_SYCL
    // Receiver's "my deps are done, my buffer is ready" event, and the sender's
    // resulting copy event. Both are published under the mutex.
    sycl::event recv_ready, copy_event;
#endif // CCL_ENABLE_SYCL
};

// === Simplified group buffer management ===
// this entry is unique for each pair of ranks
struct group_buffer_entry {
    void *buffer_ptr = nullptr;
    bool group_discovery_phase{ true };
    std::atomic<int> copy_submitted{ 0 }, reg_done{ 0 }; //signaling counters
    int copy_counter = 1, reg_counter = 1; // checking counters
#ifdef CCL_ENABLE_SYCL
    sycl::event recv_ready, copy_event;
#endif // CCL_ENABLE_SYCL
};

class shared_resources {
private:
    std::unordered_map<int, std::mutex> resource_mutexes;
    std::unordered_map<int, int> barrier_initialized_flags;

public:
    std::unordered_map<int, rank_info_vec_t> rank_info_vec_globs;
    std::unordered_map<int, std::vector<char>> all_hostnames_raw_globs;
    std::unordered_map<int, std::unordered_map<int, std::vector<void *>>> hash_table;
    // === Non-group pt2pt rendezvous, keyed per (global_id, src, dst) ===
    // unordered_map is node-based, so references handed out by
    // get_pt2pt_rendezvous() stay valid across later insertions; only the
    // lookup/insert itself needs the mutex.
    std::unordered_map<uint64_t, pt2pt_rendezvous_entry> pt2pt_rendezvous;
    std::mutex pt2pt_rendezvous_mutex;
    // === Simplified group buffer management ===
    std::vector<std::vector<std::vector<group_buffer_entry>>> group_buffers;
    std::mutex allocation_mutex;
    std::mutex barrier_waits_mutex;
    std::mutex rank_info_mutex;
    std::mutex hostnames_mutex;

    // Set by initExt() on one thread and read by others, so atomic.
    std::atomic<bool> is_multi_thread_instance{ false };
    // int current_global_id = ccl_comm::invalid_id;
    std::unordered_map<int, std::unique_ptr<pthread_barrier_t>> barrier_waits;

    shared_resources() = default;
    shared_resources(const shared_resources &) = delete;
    shared_resources &operator=(const shared_resources &) = delete;

    ~shared_resources();

    // Rendezvous slot for one pt2pt direction: src -> dst within `global_id`.
    // Both peers of a pair derive the same key, and distinct pairs get distinct
    // keys, so concurrent pt2pt operations on one communicator cannot alias.
    pt2pt_rendezvous_entry &get_pt2pt_rendezvous(int global_id, int src, int dst);

    void resize_rank_info_vec_glob(size_t new_size, int global_id);

    void resize_all_hostnames_raw(size_t new_size, int global_id);

    std::mutex &get_resource_mutex(int global_id);

    std::vector<char> &get_all_hostnames_raw_glob(int global_id);

    rank_info_vec_t &get_rank_info_vec_glob(int global_id);

    void init_barrier_wait(int threads_num, int global_id);

    pthread_barrier_t *get_barrier(int global_id);

    int get_node_rank(int ranks[2], int pair_comm_size);
};

} // namespace ccl
