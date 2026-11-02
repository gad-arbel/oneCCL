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

#include "thread_context.h"

#include <cstdlib>
#include <iostream>

namespace oneccl_test {

ThreadContext::ThreadContext(TestEnv &env,
                             const std::vector<onecclComm_t> &comms)
    : rank_(env.world_rank()), world_size_(env.world_size()),
      comm_ranks_(env.comm_ranks()),
      queue_(env.has_device() ? &env.queue() : nullptr), comms_(comms) {}

sycl::queue &ThreadContext::queue() {
    if (queue_ == nullptr) {
        std::cerr << "ThreadContext::queue() on a run with no device\n";
        std::abort();
    }
    return *queue_;
}

} // namespace oneccl_test
