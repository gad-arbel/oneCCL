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

#include "config_backend.h"
#include <optional>
#include <string>

namespace oneccl_test {

// POSIX environ implementation of the seam: setenv(..., 1), unsetenv and
// getenv, one name at a time. It never copies environ, because libccl writes
// its own defaults there during init, after the fixture is built, and a
// restore against a whole-environ copy would delete them.
class EnvironConfigBackend : public IConfigBackend {
  public:
    bool set(const std::string &name, const std::string &value) override;
    bool unset(const std::string &name) override;
    std::optional<std::string> get(const std::string &name) const override;
};

} // namespace oneccl_test
