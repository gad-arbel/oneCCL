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

#include <optional>
#include <string>

namespace oneccl_test {

// The seam ConfigScope reaches configuration through, so the scope does not
// hard-code setenv. The seam is per-name: nothing above it ever copies or
// replaces the whole scope.
//
// Not thread-safe, and not made so by a lock: every method touches
// process-global state (the global environment variables, or a similar
// backend store), and setenv / unsetenv reallocate or shift that array.
// Call it from the main thread only, while no other thread in the process can
// call getenv. That includes threads the library and its runtimes own, not
// only the test's own workers.
class IConfigBackend {
  public:
    virtual ~IConfigBackend() = default;

    // Write one entry, overwriting whatever is there. Writing the empty string
    // still counts as present: get() must then return optional(""), not
    // nullopt. set() and unset() return false on failure, so ConfigScope's
    // destructor can ADD_FAILURE() on each name without the backend depending
    // on gtest. False means nothing was written. Neither throws.
    virtual bool set(const std::string &name, const std::string &value) = 0;

    // Return one entry to its not-explicitly-set state. A registry backend
    // implements this as clearing the item, not as writing its default.
    virtual bool unset(const std::string &name) = 0;

    // Read one entry. Absent is nullopt, which distinguishes an unset name
    // from a name explicitly set to the empty string.
    virtual std::optional<std::string> get(const std::string &name) const = 0;
};

} // namespace oneccl_test
