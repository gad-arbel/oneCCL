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

#include "../data_spec.h"
#include <cstddef>

namespace oneccl_test {

// Abstract base for test data patterns.
// Contract: values are finite, non-negative for all ranks and indices.
class IDataPattern {
  public:
    virtual ~IDataPattern() = default;

    // Fill out_values[0..count) with this rank's values starting at
    // begin_index. out_values must have space for count doubles.
    virtual void fill_values(int rank, size_t begin_index, size_t count,
                             double *out_values) const = 0;

    // Pattern name for diagnostics
    virtual const char *name() const = 0;

    // Maximum absolute value this pattern emits for the given spec.
    // Used by is_supported() overflow refusals.
    static double max_absolute_value(const DataSpec &spec);
};

} // namespace oneccl_test
