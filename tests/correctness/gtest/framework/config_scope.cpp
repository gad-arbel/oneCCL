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

#include "config_scope.h"
#include "environ_config_backend.h"
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace oneccl_test {

// How much of a value an error message quotes. A longer value is cut, and the
// message gives its full length instead.
static constexpr std::size_t kMaxQuotedValueLength = 64;

static std::string quoted_value(const std::string &value) {
    if (value.size() <= kMaxQuotedValueLength) {
        return "'" + value + "'";
    }
    return "'" + value.substr(0, kMaxQuotedValueLength) + "...' (" +
           std::to_string(value.size()) + " bytes)";
}

ConfigScope::ConfigScope()
    : ConfigScope(std::make_unique<EnvironConfigBackend>()) {}

ConfigScope::ConfigScope(std::unique_ptr<IConfigBackend> backend)
    : backend_(std::move(backend)) {
    if (backend_ == nullptr) {
        throw std::invalid_argument("ConfigScope needs a backend, got null");
    }
}

ConfigScope::~ConfigScope() noexcept(false) {
    // Restore every entry before reporting any, because ADD_FAILURE() throws
    // under --gtest_throw_on_failure and would end the replay early.
    std::vector<std::string> not_restored;
    for (auto entry = undo_log_.rbegin(); entry != undo_log_.rend(); ++entry) {
        if (!restore(*entry)) {
            not_restored.push_back(entry->name);
        }
    }
    for (const std::string &name : not_restored) {
        ADD_FAILURE() << "Configuration '" << name
                      << "' could not be restored when the scope exited, so it "
                         "still carries whatever this test left in it";
    }
}

bool ConfigScope::restore(const UndoEntry &entry) {
    // Unset where the name was absent, set back otherwise.
    const bool written = entry.value.has_value()
                             ? backend_->set(entry.name, *entry.value)
                             : backend_->unset(entry.name);
    return written && backend_->get(entry.name) == entry.value;
}

void ConfigScope::set(const std::string &name, const std::string &value) {
    // Read the prior value before the write overwrites it, and log it before
    // the write, so a set() that throws below still leaves its write undoable.
    const bool already_logged = std::any_of(
        undo_log_.begin(), undo_log_.end(),
        [&name](const UndoEntry &entry) { return entry.name == name; });
    if (!already_logged) {
        undo_log_.push_back(UndoEntry{name, backend_->get(name)});
    }
    // Throws rather than using a gtest assertion: a fatal ASSERT would only
    // return from set(), leaving the caller to run under configuration that
    // never took effect.
    if (!backend_->set(name, value)) {
        // A rejected write changed nothing, so there is nothing to undo, and
        // replaying the entry would report the same name a second time.
        if (!already_logged) {
            undo_log_.pop_back();
        }
        throw std::logic_error("Configuration '" + name +
                               "' was not written: the backend rejected the "
                               "write of " +
                               quoted_value(value));
    }
    // Verifies the write actually landed (a backend could silently drop it).
    const std::optional<std::string> written = backend_->get(name);
    if (!written.has_value()) {
        throw std::logic_error("Configuration '" + name +
                               "' is absent after set(), so the backend "
                               "dropped the write of " +
                               quoted_value(value));
    }
    if (*written != value) {
        throw std::logic_error(
            "Configuration '" + name +
            "' did not round-trip through the backend after set(): wrote " +
            quoted_value(value) + ", read back " + quoted_value(*written));
    }
}

void ConfigScope::set_fallback(const std::string &name,
                               const std::string &value) {
    // Only write when the name is not already set, so an existing value wins.
    // A skipped write is not logged, because there is nothing to undo.
    if (!backend_->get(name).has_value()) {
        set(name, value);
    }
}

std::string ConfigScope::get_or_empty(const std::string &name) const {
    return backend_->get(name).value_or(std::string());
}

} // namespace oneccl_test
