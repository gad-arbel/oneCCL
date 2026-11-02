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

#include <gtest/gtest.h>

namespace oneccl_test {

// One listener, two hooks. Fatal and non-fatal failures need opposite
// treatments under MPI:
//   - OnTestPartResult: MPI_Abort on a fatal failure on the main thread, so a
//     rank that left the collective sequence tears the job down at the
//     assertion rather than hanging to ctest's timeout.
//   - OnTestEnd: MPI_Allreduce(MAX) of one bit, so one rank's non-fatal
//     failure fails the test on every rank.
class AbortListener : public ::testing::EmptyTestEventListener {
  public:
    // Whether MPI is initialized and not yet finalized.
    using MpiUsableProbe = bool (*)();
    static bool mpi_is_usable();

    // A test passes its own probe to reach the _Exit path without finalizing
    // MPI in a forked child.
    explicit AbortListener(MpiUsableProbe mpi_usable = &mpi_is_usable)
        : mpi_usable_(mpi_usable) {}

    void OnTestPartResult(const ::testing::TestPartResult &result) override;
    void OnTestEnd(const ::testing::TestInfo &info) override;

    // The process exit code OnTestPartResult drives on a fatal failure, either
    // via MPI_Abort or _Exit. Public so a death test can check for it directly
    // instead of duplicating the literal.
    static const int kAbortExitCode;

  private:
    MpiUsableProbe mpi_usable_;
};

} // namespace oneccl_test
