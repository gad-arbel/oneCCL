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

#include "test_env.h"

#include "abort_listener.h"

#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

namespace oneccl_test {

// Singleton instance of the test environment.
std::unique_ptr<TestEnv> TestEnv::instance_;

namespace {

// A failure inside create() is a message and a non-zero exit. MPI_Abort, not
// std::abort: a peer blocked in a later collective of create() is torn down
// too.
[[noreturn]] void abort_job(const std::string &message) {
    std::cerr << message << '\n';
    MPI_Abort(MPI_COMM_WORLD, AbortListener::kAbortExitCode);
    std::_Exit(AbortListener::kAbortExitCode);
}

// The devices of the first Level-Zero platform, the one onecclSetDevice
// indexes. Empty where there is none (a CPU-plugin run).
std::vector<sycl::device> level_zero_devices() {
    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            return platform.get_devices();
        }
    }
    return {};
}

sycl::queue make_in_order_queue(const sycl::device &device) {
    return sycl::queue(device, {sycl::property::queue::in_order{}});
}

// MIN-reduces capabilities_ over the world, so every rank expands an
// identical table by construction.
DeviceCapabilities min_reduce_capabilities(const DeviceCapabilities &local) {
    constexpr int kFieldCount = 3;
    int values[kFieldCount] = {local.supports_fp64 ? 1 : 0,
                               local.supports_bf16 ? 1 : 0,
                               local.devices_per_node};
    int reduced[kFieldCount] = {0, 0, 0};
    MPI_Allreduce(values, reduced, kFieldCount, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    return DeviceCapabilities{reduced[0] != 0, reduced[1] != 0, reduced[2]};
}

// Same MIN-reduce, for topology_.
LaunchTopology min_reduce_topology(const LaunchTopology &local) {
    constexpr int kFieldCount = 2;
    int values[kFieldCount] = {local.ranks_per_node, local.nodes};
    int reduced[kFieldCount] = {0, 0};
    MPI_Allreduce(values, reduced, kFieldCount, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    return LaunchTopology{reduced[0], reduced[1]};
}

} // namespace

void TestEnv::create() {
    // Create the singleton instance of the test environment.
    instance_ = std::unique_ptr<TestEnv>(new TestEnv());
    TestEnv &env = *instance_;

    // Initialize the world rank and size for the test environment.
    MPI_Comm_rank(MPI_COMM_WORLD, &env.world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &env.world_size_);

    // Node-local rank and ranks_per_node, from splitting MPI_COMM_WORLD by
    // MPI_COMM_TYPE_SHARED — each rank's shared-memory domain, i.e. its node.
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &env.local_rank_);
    int ranks_per_node = 0;
    MPI_Comm_size(local_comm, &ranks_per_node);
    MPI_Comm_free(&local_comm);

    // The communicator's rank list. The shared communicators span the whole
    // world, so a rank's position equals its world rank.
    env.comm_ranks_.resize(static_cast<size_t>(env.world_size_));
    for (int rank = 0; rank < env.world_size_; ++rank) {
        env.comm_ranks_[static_cast<size_t>(rank)] = rank;
    }

    // Device pick. A non-success code means "no device" (CPU plugin): leave
    // the optionals empty and continue. The seam reads the code, not the guard
    // behind it. The plugin does not bound the index, so the first call asks
    // for device 0, which exists whenever a device is visible. Only a plugin
    // that takes it owes the rank a device of its own, so a CPU plugin may run
    // more ranks than the node has devices.
    const std::vector<sycl::device> devices = level_zero_devices();
    const uint32_t probe_index =
        devices.empty() ? static_cast<uint32_t>(env.local_rank_) : 0;
    if (onecclSetDevice(probe_index) == onecclSuccess) {
        // The plugin took a device, so a host-backed run here would hand host
        // memory to a GPU plugin.
        if (devices.empty()) {
            abort_job("onecclSetDevice succeeded but no Level-Zero device "
                      "is visible");
        }
        if (static_cast<size_t>(env.local_rank_) >= devices.size()) {
            abort_job("local rank " + std::to_string(env.local_rank_) +
                      " has no Level-Zero device: the node exposes " +
                      std::to_string(devices.size()));
        }
        if (onecclSetDevice(static_cast<uint32_t>(env.local_rank_)) !=
            onecclSuccess) {
            abort_job("onecclSetDevice failed for local rank " +
                      std::to_string(env.local_rank_));
        }
        env.device_ = devices[static_cast<size_t>(env.local_rank_)];
        env.queue_ = make_in_order_queue(*env.device_);
    }

    // DeviceCapabilities: what the platform supports. A zeroed struct is a
    // legitimate CPU run.
    if (env.device_) {
        env.capabilities_.supports_fp64 = env.device_->has(sycl::aspect::fp64);
        // The toolchain exposes no bf16 aspect, and bf16 is supported by every
        // Intel GPU oneCCL targets (PVC, ARC, BMG); the benchmark suite assumes
        // it unconditionally. Device presence is the proxy until an aspect
        // exists to measure it against.
        env.capabilities_.supports_bf16 = true;
        env.capabilities_.devices_per_node = static_cast<int>(devices.size());
    }

    // Each node has exactly one rank with local_rank_ == 0, so summing that
    // indicator over MPI_COMM_WORLD gives the number of nodes in the job.
    int is_node_leader = (env.local_rank_ == 0) ? 1 : 0;
    int node_count = 0;
    MPI_Allreduce(&is_node_leader, &node_count, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    env.topology_.ranks_per_node = ranks_per_node;
    env.topology_.nodes = node_count;

    // Min-reduce the device capabilities across all ranks.
    env.capabilities_ = min_reduce_capabilities(env.capabilities_);

    // Topology: how ranks are distributed across nodes. Min-reduce across all
    // ranks.
    env.topology_ = min_reduce_topology(env.topology_);

    // The matrix expands against ranks_per_node x nodes. An uneven layout
    // (2 ranks on one node, 1 on another) would expand a table for a smaller
    // world than the one that runs it. Every rank holds the same reduced
    // values, so every rank takes this branch or none does.
    if (env.topology_.ranks_per_node * env.topology_.nodes != env.world_size_) {
        abort_job("uneven launch: ranks_per_node=" +
                  std::to_string(env.topology_.ranks_per_node) +
                  " x nodes=" + std::to_string(env.topology_.nodes) +
                  " != world_size=" + std::to_string(env.world_size_));
    }

    // The process's id, for the shared communicators.
    env.unique_id_ = env.mint_unique_id();
}

onecclUniqueId TestEnv::mint_unique_id() const {
    onecclUniqueId id{};
    if (world_rank_ == 0) {
        const onecclResult_t code = onecclGetUniqueId(&id);
        if (code != onecclSuccess) {
            abort_job("onecclGetUniqueId failed with code " +
                      std::to_string(static_cast<int>(code)));
        }
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return id;
}

onecclResult_t TestEnv::init_world_comm(onecclComm_t *comm,
                                        const onecclUniqueId &id,
                                        const onecclConfig_t *config) const {
    if (config != nullptr) {
        return onecclCommInitRankConfig(comm, static_cast<size_t>(world_size_),
                                        id, world_rank_, config);
    }
    return onecclCommInitRank(comm, static_cast<size_t>(world_size_), id,
                              world_rank_);
}

TestEnv &TestEnv::instance() {
    if (!instance_) {
        std::cerr << "TestEnv::instance() called before create()\n";
        std::abort();
    }
    return *instance_;
}

const sycl::device &TestEnv::device() const {
    if (!device_) {
        std::cerr << "TestEnv::device() on a run with no device\n";
        std::abort();
    }
    return *device_;
}

sycl::queue &TestEnv::queue() {
    if (!queue_) {
        std::cerr << "TestEnv::queue() on a run with no device\n";
        std::abort();
    }
    return *queue_;
}

sycl::queue TestEnv::make_queue() const {
    return make_in_order_queue(device());
}

void *TestEnv::stream() { return queue_ ? &*queue_ : nullptr; }

void TestEnv::flush() {
    if (queue_) {
        queue_->wait();
    }
}

} // namespace oneccl_test
