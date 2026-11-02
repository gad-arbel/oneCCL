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

namespace oneccl_test {

// AllReduce: all ranks contribute to every output element, in place is the
// degenerate send == recv case.
class AllReduceExecutor : public CollectiveExecutor {
  public:
    AllReduceExecutor(const CollectiveParams &params, ThreadContext &ctx);

    void setup() override;
    onecclResult_t execute(onecclComm_t comm) override;
    SourceMap source_map() const override;

  private:
    size_t count_; // Number of elements in the collective
};

} // namespace oneccl_test
