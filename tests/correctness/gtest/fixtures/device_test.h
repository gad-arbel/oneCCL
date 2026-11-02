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

#include "framework/test_env.h"
#include "framework/thread_context.h"
#include "framework/thread_groups.h"
#include "no_device_test.h"

#include "oneapi/ccl.h"
#include <functional>
#include <string>
#include <vector>

namespace oneccl_test {

// Adds the communicator and the three virtuals whose defaults leave every
// existing suite unchanged. The default bring_up adopts the process's shared
// communicators and asserts the world they carry; a suite that needs its own
// set overrides bring_up.
class DeviceTest : public NoDeviceTest {
  protected:
    void bring_up() override;
    void TearDown() override;

    TestEnv &env() { return TestEnv::instance(); }
    onecclComm_t comm() { return comms_[0]; }
    const std::vector<onecclComm_t> &comms() { return comms_; }

    // Number of communicators this fixture should create and manage.
    virtual int num_comms() const { return 1; }

    // Configuration for the communicators this fixture should create. A null
    // pointer indicates the default configuration.
    virtual const onecclConfig_t *comm_config() const { return nullptr; }

    // Whether the fixture should finalize the communicators it created in its
    // TearDown. The default is true.
    virtual bool finalize_in_teardown() const { return true; }

    // Number of test threads per rank. Only 1 is supported.
    virtual int thread_count() const { return 1; }

    // How the threads are partitioned into worlds. Unused at one thread.
    virtual ThreadGroups thread_groups() const {
        return single_thread_group(thread_count());
    }

    // The only place the tree makes a ThreadContext for a test. At one
    // thread: one context that adopts TestEnv's queue and this fixture's
    // communicators, body called on the main thread, its bool asserted. No
    // thread is spawned and no communicator is made. N > 1 fails.
    using ThreadBody = std::function<bool(ThreadContext &)>;
    void run_threads(const ThreadBody &body) {
        run_threads(thread_count(), body);
    }
    void run_threads(int n, const ThreadBody &body);

    // True when every rank passes true. Rank-collective: every rank must call
    // it.
    bool unanimous_agreement(bool local);

    // GTEST_SKIP for a capability only the loaded plugin can answer. The
    // capability goes through unanimous_agreement() first, so every rank skips
    // or none does. GTEST_SKIP returns only from this helper, so the caller
    // writes `if (skip_if_unsupported(cap, why)) return;`. Returns true when
    // it skipped, after recording the refusal in TestEnv::runtime_refusals().
    [[nodiscard]] bool skip_if_unsupported(bool supported,
                                           const std::string &why);

    std::vector<onecclComm_t> comms_;

  private:
    // Communicators this fixture created itself (a non-default num_comms, a
    // non-null comm_config, or finalize_in_teardown() false), destroyed in this
    // fixture's TearDown. The shared set is TestRuntimeEnvironment's and is not
    // owned here.
    bool owns_comms_ = false;
};

} // namespace oneccl_test
