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

#include "params/device_capabilities.h"
#include "params/param_expander.h"

#include "oneapi/ccl.h"
#include <memory>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace oneccl_test {

// The one process-global instance. main() calls create() before
// InitGoogleTest, so every member is already filled by the time any
// test runs.
//
// test_env.h and thread_context.h are the only framework/ headers that include
// SYCL. The rest of the folder is SYCL-free, and config_scope and
// environ_config_backend link into the no-device library.
class TestEnv {
  public:
    // A test skipped for a capability only the loaded plugin can report. Kept
    // so the skip stays counted and explained, like an expansion-time refusal.
    struct RuntimeRefusal {
        std::string test;
        std::string why;
    };

    // In main, before InitGoogleTest: world split, onecclSetDevice, device,
    // queue, capabilities_, topology_, unique_id_. capabilities_/topology_ are
    // MIN-reduced across MPI_COMM_WORLD so every rank expands an identical
    // table.
    static void create();

    // Aborts if create() has not run.
    static TestEnv &instance();

    int world_rank() const { return world_rank_; }
    int world_size() const { return world_size_; }
    int local_rank() const { return local_rank_; }
    const std::vector<int> &comm_ranks() const { return comm_ranks_; }
    const onecclUniqueId &unique_id() const { return unique_id_; }

    bool has_device() const { return device_.has_value(); }

    // Precondition: has_device(). Aborts otherwise.
    const sycl::device &device() const;

    // The one owned queue, in-order. Everything the harness submits runs on
    // it. Precondition: has_device(). Aborts otherwise.
    sycl::queue &queue();

    // A new queue on the same device and context, in-order, owned by the
    // caller. TestEnv neither stores nor flushes it. Precondition:
    // has_device().
    sycl::queue make_queue() const;

    // &*queue_ where a queue exists, nullptr where none does. The stream
    // argument every executor passes to the API.
    void *stream();

    // queue_->wait() where there is a queue, nothing where there is not.
    void flush();

    // A fresh onecclUniqueId: generated on world rank 0, broadcast over
    // MPI_COMM_WORLD. Rank-collective. An id names one communicator, so every
    // communicator a fixture creates for itself takes a fresh one.
    onecclUniqueId mint_unique_id() const;

    // onecclCommInitRank over the whole world with id, or
    // onecclCommInitRankConfig when config is not null.
    onecclResult_t init_world_comm(onecclComm_t *comm, const onecclUniqueId &id,
                                   const onecclConfig_t *config) const;

    // Accessors for the device capabilities and the launch topology.
    const DeviceCapabilities &capabilities() const { return capabilities_; }

    // The launch topology describes how ranks are distributed across nodes.
    const LaunchTopology &topology() const { return topology_; }

    // The sink the suites hand to ParamExpander::expand, so main can report
    // every row the expansion refused.
    std::vector<ParamExpander::Refusal> &expansion_refusals() {
        return expansion_refusals_;
    }

    const std::vector<ParamExpander::Refusal> &expansion_refusals() const {
        return expansion_refusals_;
    }

    // For a test that records a refusal of its own, so main does not report it.
    void clear_runtime_refusals() { runtime_refusals_.clear(); }

    void record_runtime_refusal(std::string test, std::string why) {
        runtime_refusals_.push_back({std::move(test), std::move(why)});
    }
    const std::vector<RuntimeRefusal> &runtime_refusals() const {
        return runtime_refusals_;
    }

  private:
    TestEnv() = default; // Private constructor to enforce singleton pattern.

    int world_rank_ = -1;
    int world_size_ = 0;
    int local_rank_ = -1;
    std::vector<int> comm_ranks_;
    onecclUniqueId unique_id_{};
    std::optional<sycl::device> device_;
    std::optional<sycl::queue> queue_;
    DeviceCapabilities capabilities_;
    LaunchTopology topology_;
    std::vector<RuntimeRefusal> runtime_refusals_;
    std::vector<ParamExpander::Refusal> expansion_refusals_;

    static std::unique_ptr<TestEnv> instance_;
};

} // namespace oneccl_test
