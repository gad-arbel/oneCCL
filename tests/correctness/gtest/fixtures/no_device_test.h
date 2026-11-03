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

#include "framework/config_scope.h"

#include <gtest/gtest.h>

#include <memory>

namespace oneccl_test {

// The root fixture. SetUp() is final so a derived fixture cannot forget to
// chain — it overrides configure() (phase 1, before any bring-up) or
// bring_up() (phase 2: MPI, device, communicator, streams) instead. config_ is
// declared here in the base, so it destructs last, after every derived member.
class NoDeviceTest : public ::testing::Test {
  protected:
    // ~ConfigScope() is noexcept(false) so --gtest_throw_on_failure can abort
    // on an unrestored name, but ~Test() is noexcept, so a ConfigScope member
    // cannot be destroyed implicitly. The failure is already recorded by
    // then; swallowing the throw only skips the abort.
    ~NoDeviceTest() override {
        try {
            config_.reset();
        } catch (...) {
        }
    }

    void SetUp() final {
        configure(*config_);
        bring_up();
    }

    // Phase 1: configure the test environment before any bring-up.
    virtual void configure(ConfigScope & /*config*/) {}
    // Phase 2: bring up MPI, device, communicator, streams.
    virtual void bring_up() {}

    std::unique_ptr<ConfigScope> config_ = std::make_unique<ConfigScope>();
};

} // namespace oneccl_test
