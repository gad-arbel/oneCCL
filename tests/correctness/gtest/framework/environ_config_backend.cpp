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

#include "environ_config_backend.h"
#include <cstdlib>
#include <optional>
#include <string>

namespace oneccl_test {

// setenv() rejects an empty name or one with '=', and c_str() cuts a name at
// an embedded NUL, so no such name can ever be written. getenv() would instead
// match "A=B" against an entry of A whose value starts with "B=", and "A\0B"
// against A itself.
static bool is_writable_name(const std::string &name) {
    return !name.empty() &&
           name.find_first_of(std::string("=\0", 2)) == std::string::npos;
}

bool EnvironConfigBackend::set(const std::string &name,
                               const std::string &value) {
    if (!is_writable_name(name)) {
        return false;
    }
    return setenv(name.c_str(), value.c_str(), /*overwrite=*/1) == 0;
}

bool EnvironConfigBackend::unset(const std::string &name) {
    if (!is_writable_name(name)) {
        return false;
    }
    return unsetenv(name.c_str()) == 0;
}

std::optional<std::string>
EnvironConfigBackend::get(const std::string &name) const {
    if (!is_writable_name(name)) {
        return std::nullopt;
    }
    const char *value = getenv(name.c_str());
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
}

} // namespace oneccl_test
