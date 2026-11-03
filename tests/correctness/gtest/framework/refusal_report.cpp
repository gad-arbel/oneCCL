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

#include "refusal_report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <unistd.h>

namespace oneccl_test {

namespace {

// gtest's own rule for --gtest_color: yes forces it, auto needs a terminal.
bool use_color() {
    const std::string flag = GTEST_FLAG_GET(color);
    if (flag == "yes" || flag == "true" || flag == "t" || flag == "1") {
        return true;
    }
    const char *term = std::getenv("TERM");
    return flag == "auto" && isatty(STDOUT_FILENO) != 0 && term != nullptr &&
           std::strcmp(term, "dumb") != 0;
}

std::string tag(bool color) {
    return color ? "\033[0;33m[  REFUSED ]\033[m " : "[  REFUSED ] ";
}

} // namespace

void print_refusal_report(const TestEnv &env) {
    const auto &expansion = env.expansion_refusals();
    const auto &runtime = env.runtime_refusals();
    const size_t total = expansion.size() + runtime.size();
    if (total == 0) {
        return;
    }
    const std::string refused = tag(use_color());
    std::ostringstream report;
    report << refused << total << (total == 1 ? " test" : " tests")
           << ", listed below:\n";
    for (const auto &refusal : expansion) {
        const CollectiveParams &params = refusal.params;
        report << refused << collective_name(params.collective) << ' '
               << dtype_name(params.dtype) << ' ' << op_name(params.op)
               << " count=" << params.count
               << (params.is_in_place ? " in-place" : " out-of-place") << ": "
               << refusal.why << '\n';
    }
    for (const auto &refusal : runtime) {
        report << refused << refusal.test << ": " << refusal.why << '\n';
    }

    // One write(), so another rank's output cannot split a line of the report.
    std::fflush(stdout);
    const std::string text = report.str();
    size_t written = 0;
    while (written < text.size()) {
        const ssize_t count =
            write(STDOUT_FILENO, text.data() + written, text.size() - written);
        if (count <= 0) {
            break;
        }
        written += static_cast<size_t>(count);
    }
}

} // namespace oneccl_test
