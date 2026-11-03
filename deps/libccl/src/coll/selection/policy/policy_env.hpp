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

#include <string>
#include <unordered_map>

namespace ccl {

struct knob_env;

namespace selection {

// Loads the policy at `path` once and returns its global_config CCL_* entries.
std::unordered_map<std::string, std::string> policy_env_overrides(const std::string& path);

// The fields a policy knob can set (knob_env), for the current call. After this thread's
// first policy lookup it is the thread's own copy: the startup values plus the knobs of its
// current phase. Before that, and from the first lookup once a multi-thread communicator
// exists, it is ccl::global_data::env().knobs. Which of the two it is only changes in
// restore_policy_knobs(), so all the reads of one phase agree. Other env_data fields are
// not reachable through it.
const knob_env& call_env();

} // namespace selection
} // namespace ccl
