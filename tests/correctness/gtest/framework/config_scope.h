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
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace oneccl_test {

// The one mechanism by which a test changes configuration. Each write records
// the name's prior state first, and the destructor puts back exactly those
// names, so a change cannot leak into the test that runs next, on any exit
// path including an assertion failure. A name the scope did not write - such
// as one libccl sets for itself during init - is never touched.
//
// Main thread only, with the same precondition as IConfigBackend: no other
// thread may call getenv while a scope writes or restores. In a test, write
// and restore only while no communicator exists: libccl's own threads, such as
// the key-value store server, read the environment while one does.
//
// Lifetime is one test at most, because the destructor reports through gtest.
class ConfigScope {
  public:
    // Captures nothing.
    ConfigScope();

    // Same, over an injected backend - a test double for the failure paths.
    // Throws std::invalid_argument on a null backend.
    explicit ConfigScope(std::unique_ptr<IConfigBackend> backend);

    // Replay the undo log, unconditionally, then report each name that did not
    // restore. Restoring every name before the first report matters under
    // --gtest_throw_on_failure, where ADD_FAILURE() throws, so the destructor
    // is noexcept(false).
    ~ConfigScope() noexcept(false);

    // Two scopes that write the same name must be destroyed in reverse order
    // of construction, as nested locals are. Nothing enforces it: destroying
    // the outer one first leaves the outer one's value behind. Copies and
    // moves are deleted because a duplicated undo log would replay twice.
    ConfigScope(const ConfigScope &) = delete;
    ConfigScope &operator=(const ConfigScope &) = delete;
    ConfigScope(ConfigScope &&) = delete;
    ConfigScope &operator=(ConfigScope &&) = delete;

    // The only sanctioned mutator. Logs the name's prior value on its first
    // write, writes, then reads the value back through the backend and throws
    // std::logic_error if the backend rejected the write or it did not
    // round-trip. A rejected first write is not logged.
    void set(const std::string &name, const std::string &value);

    // set() with overwrite = false: writes only where the name is absent, so a
    // fixture's fallback loses to a value a ctest entry set deliberately. Logs
    // only when it actually writes, and then checks the read-back as set()
    // does.
    void set_fallback(const std::string &name, const std::string &value);

    // The empty string when the name is absent. The name says so because the
    // backend's get() keeps absent and set-to-empty apart and this does not.
    std::string get_or_empty(const std::string &name) const;

  private:
    struct UndoEntry {
        std::string name;
        std::optional<std::string> value; // nullopt: the name was absent
    };

    // Puts one entry back and reads it back, as set() does. False when the
    // backend rejected the restore or did not keep it.
    bool restore(const UndoEntry &entry);

    std::unique_ptr<IConfigBackend> backend_;
    std::vector<UndoEntry> undo_log_; // one entry per name, at its first write
};

} // namespace oneccl_test
