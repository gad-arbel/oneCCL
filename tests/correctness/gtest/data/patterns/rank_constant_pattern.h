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

#include "data_pattern.h"

namespace oneccl_test {

// Rank-constant pattern: value = rank + buffer_index + 1
// Exactly representable in all dtypes, bit-exact comparison.
class RankConstantPattern : public IDataPattern {
  public:
    explicit RankConstantPattern(size_t buffer_index = 0)
        : buffer_index_(buffer_index) {}

    void fill_values(int rank, size_t begin_index, size_t count,
                     double *out_values) const override;
    const char *name() const override { return "RankConstant"; }

  private:
    size_t buffer_index_;
};

} // namespace oneccl_test
