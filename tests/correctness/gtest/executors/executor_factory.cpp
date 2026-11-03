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

#include "executor_factory.h"

#include "allreduce_executor.h"

namespace oneccl_test {

std::unique_ptr<CollectiveExecutor>
ExecutorFactory::create(const CollectiveParams &params, ThreadContext &ctx) {
    switch (params.collective) {
    case Collective::ONECCL_ALLREDUCE:
        return std::make_unique<AllReduceExecutor>(params, ctx);
    case Collective::ONECCL_REDUCE:
    case Collective::ONECCL_BROADCAST:
    case Collective::ONECCL_ALLGATHER:
    case Collective::ONECCL_ALLTOALL:
    case Collective::ONECCL_ALLTOALLV:
    case Collective::ONECCL_REDUCE_SCATTER:
    case Collective::ONECCL_SCATTER:
    case Collective::ONECCL_SEND_RECV:
        return nullptr;
    }
    return nullptr;
}

} // namespace oneccl_test
