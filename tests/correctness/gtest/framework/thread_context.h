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

#include "test_env.h"

#include "oneapi/ccl.h"
#include <sycl/sycl.hpp>
#include <vector>

namespace oneccl_test {

// The one object a step sees, and what an executor is constructed against in
// TestEnv's place: rank(), world_size() and stream() are the questions an
// executor would otherwise put to TestEnv. Owned by DeviceTest, one per thread.
//
// Only the adopt constructor exists: at one thread the context takes
// TestEnv's queue and the fixture's communicators, so rank() and world_size()
// are the world's and nothing is created or destroyed.
class ThreadContext {
  public:
    ThreadContext(TestEnv &env, const std::vector<onecclComm_t> &comms);

    ThreadContext(const ThreadContext &) = delete;
    ThreadContext &operator=(const ThreadContext &) = delete;

    int rank() const { return rank_; }
    // The world size. Equals comm_ranks().size() while one group spans the
    // world.
    int world_size() const { return world_size_; }

    // The communicator's rank list, which the data layer builds its spec
    // from.
    const std::vector<int> &comm_ranks() const { return comm_ranks_; }

    // A queue pointer where there is a queue, nullptr where there is not. The
    // stream argument every executor passes to the API.
    void *stream() { return queue_; }

    onecclComm_t comm() const { return comms_[0]; }
    const std::vector<onecclComm_t> &comms() const { return comms_; }

    bool has_device() const { return queue_ != nullptr; }

    // The queue buffers are allocated and copied on. Precondition:
    // has_device().
    sycl::queue &queue();

  private:
    int rank_ = -1;
    int world_size_ = 0;
    std::vector<int> comm_ranks_;
    sycl::queue *queue_ = nullptr;
    std::vector<onecclComm_t> comms_;
};

} // namespace oneccl_test
