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

// Plugin for tests/plugin_selection.cpp, selected by its path.
// ONECCL_TEST_FAKE_PLUGIN_MODE picks how it fails. It is read on each call,
// so one library serves every test.
#include "internal/api/plugin.h"
#include "oneapi/ccl.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

extern "C" onecclResult_t fake_plugin_reentry_result;

namespace {

constexpr auto kInitSleep = std::chrono::milliseconds(300);

bool mode_is(const char *mode) {
    const char *value = getenv("ONECCL_TEST_FAKE_PLUGIN_MODE");
    return value != nullptr && strcmp(value, mode) == 0;
}

onecclResult_t fake_init() {
    if (mode_is("init_fail")) {
        return onecclError;
    }
    if (mode_is("init_throw")) {
        throw std::runtime_error("fake plugin init failed");
    }
    if (mode_is("init_reenter")) {
        int version = -1;
        fake_plugin_reentry_result = onecclGetVersion(&version);
    }
    if (mode_is("init_sleep")) {
        std::this_thread::sleep_for(kInitSleep);
    }
    return onecclSuccess;
}

onecclResult_t fake_mem_alloc(void **ptr, size_t /*size*/) {
    *ptr = nullptr;
    return onecclSuccess;
}

} // namespace

extern "C" {
// Result of the API call the plugin made from its init, read by the test
onecclResult_t fake_plugin_reentry_result = onecclSuccess;

void *onecclPluginCall(onecclPluginCall_t call_type) {
    switch (call_type) {
    case ONECCL_PLUGIN_INIT:
        if (mode_is("entrypoint_throw")) {
            throw std::runtime_error("fake plugin entrypoint failed");
        }
        return reinterpret_cast<void *>(&fake_init);
    case ONECCL_PLUGIN_MEM_ALLOC:
        return reinterpret_cast<void *>(&fake_mem_alloc);
    default:
        return nullptr;
    }
}
}
