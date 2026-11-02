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

#include "mismatch_report.h"
#include <gtest/gtest.h>
#include <sstream>

namespace oneccl_test {

void MismatchReport::add_single_mismatch(size_t index, double actual,
                                         double expected, double tolerance) {
    if (recorded_mismatches_.size() < kMaxReported) {
        recorded_mismatches_.push_back({index, actual, expected, tolerance});
    }
    ++mismatch_count_;
}

void MismatchReport::fail_now_with_report(int rank) const {
    if (no_mismatch_recorded())
        return;

    std::ostringstream message;
    message << "Rank " << rank << ": " << mismatch_count_ << " mismatch(es)\n";

    for (const auto &entry : recorded_mismatches_) {
        message << "  [" << entry.index << "] actual=" << entry.actual
                << " expected=" << entry.expected
                << " tolerance=" << entry.tolerance << "\n";
    }

    if (mismatch_count_ > kMaxReported) {
        message << "  ... and " << (mismatch_count_ - kMaxReported)
                << " more\n";
    }

    ADD_FAILURE() << message.str();
}

} // namespace oneccl_test
