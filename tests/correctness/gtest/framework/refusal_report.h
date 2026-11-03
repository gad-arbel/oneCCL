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

#include "test_env.h"

namespace oneccl_test {

// Prints every row the expansion refused and every test that skipped at run
// time, each with its reason, to stdout. Every rank holds the same refusals,
// so the caller prints from one rank. The report goes out in one write(), so
// another rank's output cannot land inside a line of it.
void print_refusal_report(const TestEnv &env);

} // namespace oneccl_test
