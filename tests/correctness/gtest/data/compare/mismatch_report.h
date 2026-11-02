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

#include <cstddef>
#include <vector>

namespace oneccl_test {

// Collects mismatches for capped reporting.
// Stores first kMaxReported, then summarizes the rest.
class MismatchReport {
  public:
    static constexpr size_t kMaxReported = 10;

    struct Entry {
        size_t index;
        double actual;
        double expected;
        double tolerance;
    };

    // Add one mismatch
    void add_single_mismatch(size_t index, double actual, double expected,
                             double tolerance);

    // True when no mismatch was recorded
    bool no_mismatch_recorded() const { return mismatch_count_ == 0; }

    // Total mismatch count
    size_t mismatch_count() const { return mismatch_count_; }

    // Non-fatal failure with formatted report
    void fail_now_with_report(int rank) const;

  private:
    std::vector<Entry> recorded_mismatches_;
    size_t mismatch_count_ = 0;
};

} // namespace oneccl_test
