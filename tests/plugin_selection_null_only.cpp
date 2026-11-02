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

#include "oneapi/ccl.h"

#include <cstdlib>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <unistd.h>

namespace {

// Whether a directory in LD_LIBRARY_PATH has the file. Checked without
// dlopen(): a loaded plugin would change what the selection sees.
bool in_library_path(const std::string &name) {
    const char *library_path = getenv("LD_LIBRARY_PATH");
    std::istringstream dirs(library_path != nullptr ? library_path : "");
    std::string dir;
    while (std::getline(dirs, dir, ':')) {
        if (!dir.empty() && access((dir + "/" + name).c_str(), F_OK) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

// The null plugin does no communication: only CCL_PLUGIN=ONECCL_NULL
// selects it
//   no CCL_PLUGIN, only the null plugin in the search path
//   onecclGetVersion() -> fails
//   expect log: no plugin loaded, null plugin scored 0
TEST(PluginSelectionNullOnly, NullPluginNotSelectedAutomatically) {
    unsetenv("CCL_PLUGIN");
    setenv("CCL_LOG_LEVEL", "info", 1);
    if (in_library_path("libccl_legacy.so") ||
        in_library_path("libccl_legacy_cpu.so")) {
        GTEST_SKIP() << "a legacy plugin is in LD_LIBRARY_PATH";
    }

    testing::internal::CaptureStderr();
    int version = -1;
    EXPECT_EQ(onecclGetVersion(&version), onecclError);
    std::string const log = testing::internal::GetCapturedStderr();
    EXPECT_NE(log.find("Could not load any plugin"), std::string::npos) << log;
    EXPECT_NE(log.find("Platform score for ONECCL_NULL: 0"), std::string::npos)
        << log;
}
