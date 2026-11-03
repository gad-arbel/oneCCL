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

#include "collective_executor.h"
#include "framework/thread_context.h"
#include "params/collective_params.h"

#include <memory>

namespace oneccl_test {

class ExecutorFactory {
  public:
    static std::unique_ptr<CollectiveExecutor>
    create(const CollectiveParams &params, ThreadContext &ctx);
};

} // namespace oneccl_test
