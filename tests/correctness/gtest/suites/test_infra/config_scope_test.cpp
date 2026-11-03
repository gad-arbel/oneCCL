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

#include "framework/config_backend.h"
#include "framework/config_scope.h"
#include "framework/environ_config_backend.h"
#include <cstdlib>
#include <exception>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace oneccl_test;

// A name nothing in the repo, in a ctest entry or in a developer shell sets, so
// "absent afterwards" is always the correct assertion. The mechanism works the
// same for any name, so it behaves identically on a real CCL_* name; which
// names the library would still observe mid-suite is a separate audit, not
// this suite's subject.
static constexpr const char *kProbeName = "ONECCL_TEST_CONFIG_SCOPE_PROBE";

// A second probe, written straight through the backend where a test stands in
// for libccl writing its own configuration during init.
static constexpr const char *kLibraryWrittenName =
    "ONECCL_TEST_CONFIG_SCOPE_LIBRARY_WRITTEN";

// A third name, for the replay tests that need a name on each side of the one
// that fails.
static constexpr const char *kOtherName = "ONECCL_TEST_CONFIG_SCOPE_OTHER";

// The value names what the write is for, not which test performed it: the tests
// below are ordered by registration, so a value spelling out "written by the
// first test" would become wrong as soon as any TEST is inserted above it.
static constexpr const char *kProbeValue = "written-through-the-scope";
static constexpr const char *kLibraryValue = "written-by-the-library";

// Set by the first test of the leak pair, read by the second. The pair spans a
// test boundary on purpose - that boundary is what the scope exists to protect
// - but gtest runs tests in REGISTRATION order, so --gtest_filter,
// --gtest_shuffle and GTEST_TOTAL_SHARDS can each run the second one alone. It
// then skips instead of reporting a pass its assertions did not earn.
static bool probe_was_written_by_an_earlier_test = false;

// The copy and move deletions are compile-time only, so no runtime assertion
// can observe them: a duplicated scope would replay its undo log twice.
static_assert(!std::is_copy_constructible_v<ConfigScope>,
              "A copyable ConfigScope would replay its undo log twice");
static_assert(!std::is_copy_assignable_v<ConfigScope>,
              "A copy-assignable ConfigScope would replay its undo log twice");
static_assert(!std::is_move_constructible_v<ConfigScope>,
              "A movable ConfigScope would replay its undo log twice");
static_assert(!std::is_move_assignable_v<ConfigScope>,
              "A move-assignable ConfigScope would replay its undo log twice");

// What a FakeBackend holds and how it misbehaves. Owned by the caller, so it
// outlives the scope that destroys the backend.
struct FakeStore {
    std::map<std::string, std::string> entries;
    bool drop_writes = false;   // set() returns true but keeps nothing
    bool reject_writes = false; // set() returns false and keeps nothing
    bool mangle_writes = false; // set() returns true but keeps a wrong value
    std::string fail_set_of;    // set() of this name returns false
    std::string fail_unset_of;  // unset() of this name returns false
    int set_count = 0;
    int unset_count = 0;
};

// An in-memory backend for the paths the real environ backend cannot reach: a
// write the backend drops, rejects or mangles, a restore that fails, and a
// count of what the destructor touched.
class FakeBackend : public IConfigBackend {
  public:
    explicit FakeBackend(FakeStore *store) : store_(store) {}

    bool set(const std::string &name, const std::string &value) override {
        ++store_->set_count;
        if (store_->reject_writes || name == store_->fail_set_of) {
            return false;
        }
        if (store_->mangle_writes) {
            store_->entries[name] = value + "-mangled";
        } else if (!store_->drop_writes) {
            store_->entries[name] = value;
        }
        return true;
    }

    bool unset(const std::string &name) override {
        ++store_->unset_count;
        if (name == store_->fail_unset_of) {
            return false;
        }
        store_->entries.erase(name);
        return true;
    }

    std::optional<std::string> get(const std::string &name) const override {
        const auto entry = store_->entries.find(name);
        if (entry == store_->entries.end()) {
            return std::nullopt;
        }
        return entry->second;
    }

  private:
    FakeStore *store_;
};

// The leak pair, first half: write through the scope and see the write land,
// then write a second name straight through the backend, standing in for a
// library write at init.
// Pseudo code:
//   assert probe absent; unset library name
//   { scope.set(probe, value); assert getenv(probe) == value
//     backend.set(library name) }
//   mark the pair's first half as run
TEST(ConfigScopeTest,
     set_with_a_probe_name_then_writes_through_to_the_environment) {
    ASSERT_EQ(getenv(kProbeName), nullptr)
        << "The probe must start absent, or the next test's leak assertion "
           "means nothing; leaked value: "
        << getenv(kProbeName);
    // Only the second test of the pair removes this name, so a filtered or
    // repeated run can arrive here with it still set. Start from absent.
    ASSERT_TRUE(EnvironConfigBackend().unset(kLibraryWrittenName));

    {
        ConfigScope config;
        config.set(kProbeName, kProbeValue);

        EXPECT_EQ(config.get_or_empty(kProbeName), kProbeValue)
            << "The value did not read back through the scope";
        ASSERT_NE(getenv(kProbeName), nullptr)
            << "set() did not reach the environment";
        EXPECT_STREQ(getenv(kProbeName), kProbeValue)
            << "set() wrote a different value than it was given";

        ASSERT_TRUE(
            EnvironConfigBackend().set(kLibraryWrittenName, kLibraryValue));
    }

    // Only now: the writes happened and the scope has closed, so the next
    // test's assertions have something to observe.
    probe_was_written_by_an_earlier_test = true;
}

// The leak pair, second half: a configuration change made by one test is not
// visible to the test that runs after it, and a name the library wrote for
// itself is left in place.
// Pseudo code:
//   skip unless the first half ran just before
//   assert probe absent
//   assert library name still holds the library value; unset it
TEST(
    ConfigScopeTest,
    destructor_with_a_prior_test_write_then_leaks_nothing_across_the_test_boundary) {
    if (!probe_was_written_by_an_earlier_test) {
        GTEST_SKIP()
            << "set_with_a_probe_name_then_writes_through_to_the_environment "
               "did not run before this "
               "test, so the assertions below would pass with nothing to "
               "check - there is no leak to observe. Run the whole suite, "
               "unfiltered, unsharded and unshuffled, to test the cross-test "
               "boundary.";
    }
    // Consume the flag, so a --gtest_repeat iteration that runs this test
    // before the first one skips instead of reading the last iteration's flag.
    probe_was_written_by_an_earlier_test = false;

    EXPECT_EQ(getenv(kProbeName), nullptr)
        << "The previous test's configuration change leaked out of it; leaked "
           "value: "
        << getenv(kProbeName);

    const std::optional<std::string> library_value =
        EnvironConfigBackend().get(kLibraryWrittenName);
    // The previous test cannot clean this up itself without defeating the
    // check, so this one does.
    EXPECT_TRUE(EnvironConfigBackend().unset(kLibraryWrittenName));
    ASSERT_TRUE(library_value.has_value())
        << "Restore removed a name the scope never wrote, so it would delete "
           "libccl's own init-time configuration";
    EXPECT_EQ(*library_value, kLibraryValue);
}

// The prior-value branch of the replay, which the leak pair never reaches: the
// pair only covers a name that was absent. This and the tests below wrap their
// bodies in an outer scope, so each restores whatever it dirties on any outcome
// and cannot leak into a later test.
// Pseudo code:
//   outer.set(probe, "before")
//   { inner.set(probe, "after") }
//   assert probe == "before"
TEST(ConfigScopeTest,
     destructor_with_an_already_present_name_then_restores_its_prior_value) {
    ConfigScope outer;
    outer.set(kProbeName, "before");

    {
        ConfigScope inner;
        inner.set(kProbeName, "after");
        EXPECT_EQ(inner.get_or_empty(kProbeName), "after");
    }

    EXPECT_EQ(outer.get_or_empty(kProbeName), "before")
        << "Restore reset an already-present name to the wrong value, or "
           "removed it entirely";
}

// A second write of a name inside one scope does not change what the scope
// restores.
// destructor_with_repeated_writes_of_a_name_then_touches_each_name_once checks
// the log itself. Pseudo code:
//   outer.set(probe, "before")
//   { inner.set(probe, "first"); inner.set(probe, "second") }
//   assert probe == "before"
TEST(
    ConfigScopeTest,
    destructor_with_two_writes_of_a_name_then_restores_the_value_before_the_first) {
    ConfigScope outer;
    outer.set(kProbeName, "before");

    {
        ConfigScope inner;
        inner.set(kProbeName, "first");
        inner.set(kProbeName, "second");
    }

    EXPECT_EQ(outer.get_or_empty(kProbeName), "before")
        << "Restore put back an intermediate value rather than the one the "
           "name held before the scope first wrote it";
}

// The scope restores on every exit path "including an assertion failure", which
// is its most-advertised property and the one no other test here exercises: a
// destructor made conditional on !::testing::Test::HasFailure() stays green
// through every other test in this file.
//
// The failure has to be a REAL recorded one for HasFailure() to see it, so this
// cannot use EXPECT_NONFATAL_FAILURE - that diverts the failure into an
// interceptor and leaves HasFailure() false, which is exactly how such a
// destructor slips through. It runs in a forked child instead: the child fails
// for real, and reports through its exit status whether the scope still
// restored.
// Pseudo code:
//   in a child: { scope.set(probe); ADD_FAILURE() }
//               exit 0 if probe is gone, else 1
//   assert the child exited 0
TEST(ConfigScopeDeathTest,
     destructor_with_an_already_failed_test_then_still_restores) {
    EXPECT_EXIT(
        {
            FakeStore store;
            {
                ConfigScope config(std::make_unique<FakeBackend>(&store));
                config.set(kProbeName, kProbeValue);
                ADD_FAILURE() << "deliberate failure, so this scope exits the "
                                 "way it would under a real failed assertion";
            }
            std::exit(store.entries.count(kProbeName) == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "")
        << "The scope did not restore after the test had already failed, so a "
           "failing test leaves its configuration behind for the next one";
}

// Under --gtest_throw_on_failure, ADD_FAILURE() throws. The destructor still
// restores every name before it reports the first failure, and the exception
// reaches the caller instead of calling std::terminate. Runs in a forked child,
// because the flag and the deliberate failure must not reach the parent.
// Pseudo code:
//   in a child: set --gtest_throw_on_failure
//               try { scope.set(probe); scope.set(failing name) } catch
//               exit 0 if probe is gone, else 1
//   assert the child exited 0
TEST(ConfigScopeDeathTest,
     destructor_with_a_throwing_failure_report_then_restores_every_name_first) {
    EXPECT_EXIT(
        {
            GTEST_FLAG_SET(throw_on_failure, true);
            FakeStore store;
            store.fail_unset_of = kLibraryWrittenName;
            try {
                ConfigScope config(std::make_unique<FakeBackend>(&store));
                config.set(kProbeName, kProbeValue);
                config.set(kLibraryWrittenName, kProbeValue);
            } catch (const std::exception &) {
            }
            std::exit(store.entries.count(kProbeName) == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "")
        << "A throwing failure report ended the replay, or terminated the "
           "process, before every name was restored";
}

// A name that cannot be restored is reported by name, and the replay goes on to
// restore the rest rather than stopping at the first failure. The failing name
// sits between two others, so a replay that stops leaks one of them in either
// replay order.
// Pseudo code:
//   unset() of the library name fails
//   { scope.set(probe); scope.set(library name); scope.set(other) }
//   assert one failure that names the library name
//   assert probe and other are both gone
TEST(
    ConfigScopeTest,
    destructor_with_a_failed_unset_then_reports_the_name_and_restores_the_rest) {
    FakeStore store;
    store.fail_unset_of = kLibraryWrittenName;

    EXPECT_NONFATAL_FAILURE(
        {
            ConfigScope config(std::make_unique<FakeBackend>(&store));
            config.set(kProbeName, kProbeValue);
            config.set(kLibraryWrittenName, kProbeValue);
            config.set(kOtherName, kProbeValue);
        },
        kLibraryWrittenName);

    EXPECT_EQ(store.entries.count(kProbeName), 0u)
        << "One failed restore stopped the replay, so an earlier name leaked";
    EXPECT_EQ(store.entries.count(kOtherName), 0u)
        << "One failed restore stopped the replay, so a later name leaked";
}

// The other restore branch: setting a prior value back can fail too.
// Pseudo code:
//   probe = "before"
//   { scope.set(probe, "after"); make set() of probe fail }
//   assert one failure that names probe
TEST(ConfigScopeTest,
     destructor_with_a_failed_set_back_of_a_prior_value_then_reports_the_name) {
    FakeStore store;
    store.entries[kProbeName] = "before";

    EXPECT_NONFATAL_FAILURE(
        {
            ConfigScope config(std::make_unique<FakeBackend>(&store));
            config.set(kProbeName, "after");
            store.fail_set_of = kProbeName;
        },
        kProbeName);
}

// A restore is read back, as set() is: a set-back the backend accepts but does
// not keep is reported, not trusted.
// Pseudo code:
//   probe = "before"
//   { scope.set(probe, value); make the backend mangle writes }
//   assert one failure that names probe
TEST(
    ConfigScopeTest,
    destructor_with_a_set_back_the_backend_did_not_keep_then_reports_the_name) {
    FakeStore store;
    store.entries[kProbeName] = "before";

    EXPECT_NONFATAL_FAILURE(
        {
            ConfigScope config(std::make_unique<FakeBackend>(&store));
            config.set(kProbeName, kProbeValue);
            store.mangle_writes = true;
        },
        kProbeName);
}

// Each name is logged once however often it is written, by set() or
// set_fallback(), so the destructor touches each name exactly once.
// Pseudo code:
//   { scope.set_fallback(probe); scope.set(probe) twice
//     scope.set(library name) }
//   assert 4 sets and 2 unsets in total, and the store is empty
TEST(ConfigScopeTest,
     destructor_with_repeated_writes_of_a_name_then_touches_each_name_once) {
    FakeStore store;
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set_fallback(kProbeName, "fallback");
        config.set(kProbeName, "second");
        config.set(kProbeName, "third");
        config.set(kLibraryWrittenName, kProbeValue);
    }

    EXPECT_EQ(store.set_count, 4) << "The destructor wrote a name back";
    EXPECT_EQ(store.unset_count, 2)
        << "The destructor replayed a name more than once";
    EXPECT_TRUE(store.entries.empty());
}

// A prior empty string is a present value: restore sets it back rather than
// unsetting the name.
// Pseudo code:
//   probe = ""
//   { scope.set(probe, value) }
//   assert probe is present and == "", with no unset
TEST(ConfigScopeTest,
     destructor_with_a_prior_empty_string_then_sets_it_back_as_present) {
    FakeStore store;
    store.entries[kProbeName] = "";
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set(kProbeName, kProbeValue);
    }

    ASSERT_EQ(store.entries.count(kProbeName), 1u)
        << "Restore unset a name that was present and set to the empty string";
    EXPECT_EQ(store.entries[kProbeName], "");
    EXPECT_EQ(store.unset_count, 0);
}

// The leak pair checks this too, but skips when run alone. Here a name the
// scope never wrote stays, and only the written name is restored.
// Pseudo code:
//   library name = library value
//   { scope.set(probe) }
//   assert library name unchanged, 1 set and 1 unset
TEST(ConfigScopeTest,
     destructor_with_a_name_the_scope_did_not_write_then_leaves_it_alone) {
    FakeStore store;
    store.entries[kLibraryWrittenName] = kLibraryValue;
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set(kProbeName, kProbeValue);
    }

    EXPECT_EQ(store.entries[kLibraryWrittenName], kLibraryValue);
    EXPECT_EQ(store.unset_count, 1);
    EXPECT_EQ(store.set_count, 1);
}

// Restore puts back the state before the scope's first write, without
// comparing against the current value. So when the library overwrites a name
// the scope also wrote (libccl rewrites FI_PROVIDER with overwrite = 1), the
// library's value is replaced on the way out.
// Pseudo code:
//   probe = "before"
//   { scope.set(probe, value); library overwrites probe }
//   assert probe == "before"
TEST(
    ConfigScopeTest,
    destructor_with_a_value_written_after_the_scope_wrote_then_replaces_it_with_the_prior_value) {
    FakeStore store;
    store.entries[kProbeName] = "before";
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set(kProbeName, kProbeValue);
        store.entries[kProbeName] = kLibraryValue;
    }

    EXPECT_EQ(store.entries[kProbeName], "before");
}

// A null backend is refused where it is passed, not at the first use.
// Pseudo code:
//   assert ConfigScope(null) throws invalid_argument
TEST(ConfigScopeTest, constructor_with_a_null_backend_then_throws) {
    EXPECT_THROW(ConfigScope(std::unique_ptr<IConfigBackend>()),
                 std::invalid_argument);
}

// set() reads back through the backend and throws when the write did not
// round-trip. Nothing above can reach that path, because the environment
// variable backend accepts every legal write; the fake drops writes the way a
// stricter backend might reject an unknown name.
// Pseudo code:
//   backend drops writes
//   assert set(probe, value) throws
//   assert set(probe, "") throws
TEST(ConfigScopeTest, set_with_a_backend_that_drops_the_write_then_throws) {
    FakeStore store;
    store.drop_writes = true;
    ConfigScope config(std::make_unique<FakeBackend>(&store));

    EXPECT_THROW(config.set(kProbeName, kProbeValue), std::logic_error)
        << "set() accepted a write the backend did not keep, so a test would "
           "run on under configuration that never landed";

    // The empty string is the case a read-back written as
    // value_or(std::string()) gets wrong: absent must not read as set-to-empty,
    // or a dropped write of "" passes.
    EXPECT_THROW(config.set(kProbeName, ""), std::logic_error)
        << "set() read an absent name as the empty string it had just tried to "
           "write";
}

// The backend's own failure report, separate from the read-back. The name
// already holds the value, so only the report can make set() throw. A
// rejected write changed nothing, so the destructor has nothing to restore.
// Pseudo code:
//   probe = value; backend rejects writes
//   { assert scope.set(probe, value) throws }
//   assert 1 set and no unset in total
TEST(
    ConfigScopeTest,
    set_with_a_backend_that_rejects_the_write_then_throws_and_restores_nothing) {
    FakeStore store;
    store.entries[kProbeName] = kProbeValue;
    store.reject_writes = true;
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        EXPECT_THROW(config.set(kProbeName, kProbeValue), std::logic_error)
            << "set() ignored a backend that reported the write as failed";
    }

    EXPECT_EQ(store.set_count, 1) << "The destructor replayed a rejected write";
    EXPECT_EQ(store.unset_count, 0);
}

// A rejected rewrite of a name the scope already logged keeps every entry. Only
// a rejected first write drops one, and only its own: dropping the newest entry
// instead would lose the name written after it.
// Pseudo code:
//   { scope.set(probe); scope.set(library name)
//     make set() of probe fail; assert scope.set(probe) throws }
//   assert the store is empty
TEST(ConfigScopeTest, set_with_a_rejected_rewrite_then_keeps_every_undo_entry) {
    FakeStore store;
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set(kProbeName, kProbeValue);
        config.set(kLibraryWrittenName, kProbeValue);
        store.fail_set_of = kProbeName;
        EXPECT_THROW(config.set(kProbeName, "rewrite"), std::logic_error);
    }

    EXPECT_EQ(store.entries.count(kProbeName), 0u)
        << "The rejected rewrite dropped the name's own undo entry";
    EXPECT_EQ(store.entries.count(kLibraryWrittenName), 0u)
        << "The rejected rewrite dropped the undo entry of a later name";
}

// A write that lands with the wrong value throws, and the destructor still
// undoes it, because the undo entry was logged before the write.
// Pseudo code:
//   probe = "before"; backend mangles writes
//   { assert scope.set(probe, value) throws; stop mangling }
//   assert probe == "before"
TEST(ConfigScopeTest,
     set_with_a_mismatched_read_back_then_throws_and_still_restores) {
    FakeStore store;
    store.entries[kProbeName] = "before";
    store.mangle_writes = true;
    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        EXPECT_THROW(config.set(kProbeName, kProbeValue), std::logic_error)
            << "set() accepted a read-back that differs from the write";
        store.mangle_writes = false;
    }

    EXPECT_EQ(store.entries[kProbeName], "before")
        << "A set() that threw left its write behind";
}

// The environ backend reports a name POSIX cannot hold: setenv() rejects '='
// and the empty name, and c_str() would cut a name at an embedded NUL. Pseudo
// code:
//   for name in "BAD=NAME", "", "BAD\0NAME":
//     assert backend.set(name) and backend.unset(name) return false
TEST(ConfigScopeTest,
     environ_backend_set_and_unset_with_an_invalid_name_then_return_false) {
    for (const std::string &name : {std::string("BAD=NAME"), std::string(),
                                    std::string("BAD\0NAME", 8)}) {
        EXPECT_FALSE(EnvironConfigBackend().set(name, kProbeValue))
            << "set() hid a failure for an invalid name";
        EXPECT_FALSE(EnvironConfigBackend().unset(name))
            << "unset() hid a failure for an invalid name";
    }
}

// getenv("A=B") matches the entry "A=B=..." of A. get() must still report a
// name that set() can never write as absent, or set_fallback() skips it
// silently instead of throwing.
// Pseudo code:
//   scope.set(probe, "NAME=value")
//   assert get(probe + "=NAME") is absent
//   assert set_fallback(probe + "=NAME") throws
//   assert get("") is absent
TEST(ConfigScopeTest,
     environ_backend_get_with_a_name_containing_equals_then_reports_absent) {
    const std::string invalid_name = std::string(kProbeName) + "=NAME";
    ConfigScope config;
    config.set(kProbeName, "NAME=value");

    EXPECT_FALSE(EnvironConfigBackend().get(invalid_name).has_value())
        << "get() read another name's value through a name with '='";
    EXPECT_THROW(config.set_fallback(invalid_name, kProbeValue),
                 std::logic_error)
        << "set_fallback() skipped a name it can never write";
    EXPECT_EQ(config.get_or_empty(invalid_name), "");
    EXPECT_FALSE(EnvironConfigBackend().get("").has_value())
        << "get() read a value through the empty name";
}

// c_str() cuts "A\0B" to "A", so without a check every call would act on A. A
// name with an embedded NUL is absent, and writing it throws instead of
// writing A.
// Pseudo code:
//   scope.set(probe, "before")
//   assert get(probe + "\0X") is absent
//   assert set() and set_fallback() of probe + "\0X" throw
//   assert probe == "before"
TEST(ConfigScopeTest,
     environ_backend_get_with_a_name_containing_a_nul_then_reports_absent) {
    std::string invalid_name = kProbeName;
    invalid_name += std::string("\0X", 2);
    ConfigScope config;
    config.set(kProbeName, "before");

    EXPECT_FALSE(EnvironConfigBackend().get(invalid_name).has_value())
        << "get() read another name's value through a name with a NUL";
    EXPECT_THROW(config.set(invalid_name, kProbeValue), std::logic_error)
        << "set() wrote another name through a name with a NUL";
    EXPECT_THROW(config.set_fallback(invalid_name, kProbeValue),
                 std::logic_error)
        << "set_fallback() skipped a name it can never write";
    EXPECT_EQ(config.get_or_empty(kProbeName), "before");
}

// set() of an invalid name throws once. Without dropping the undo entry, the
// destructor would also report the name as not restored.
// Pseudo code:
//   assert set("BAD=NAME") throws; assert set("") throws
//   the destructor reports nothing
TEST(ConfigScopeTest, set_with_an_invalid_name_then_reports_once) {
    ConfigScope config;
    EXPECT_THROW(config.set("BAD=NAME", kProbeValue), std::logic_error);
    EXPECT_THROW(config.set("", kProbeValue), std::logic_error);
}

// The same round-trip requirement on the other mutator.
// Pseudo code:
//   backend drops writes
//   assert set_fallback(probe) throws
TEST(ConfigScopeTest,
     set_fallback_with_a_backend_that_drops_the_write_then_throws) {
    FakeStore store;
    store.drop_writes = true;
    ConfigScope config(std::make_unique<FakeBackend>(&store));

    EXPECT_THROW(config.set_fallback(kProbeName, kProbeValue), std::logic_error)
        << "set_fallback() accepted a write the backend did not keep";
}

// A fallback never overwrites a value that is already present.
// Pseudo code:
//   outer.set(probe, "set-deliberately")
//   { inner.set_fallback(probe, "fallback")
//     assert probe == "set-deliberately" }
TEST(ConfigScopeTest,
     set_fallback_with_an_already_present_value_then_keeps_it) {
    ConfigScope outer;
    outer.set(kProbeName, "set-deliberately");

    {
        ConfigScope inner;
        inner.set_fallback(kProbeName, "fallback");
        EXPECT_EQ(inner.get_or_empty(kProbeName), "set-deliberately")
            << "set_fallback() overwrote a value that was already present";
    }
}

// The same, on the one prior value that is not distinguishable from "absent"
// without the seam's optional: a name a ctest entry deliberately set to the
// empty string, e.g. CCL_FOO= to disable a feature. This is the whole reason
// IConfigBackend::get returns std::optional.
// Pseudo code:
//   outer.set(probe, "")
//   { inner.set_fallback(probe, "fallback")
//     assert probe is present and == "" }
TEST(ConfigScopeTest, set_fallback_with_a_value_set_to_empty_then_keeps_it) {
    ConfigScope outer;
    outer.set(kProbeName, "");

    {
        ConfigScope inner;
        inner.set_fallback(kProbeName, "fallback");

        EXPECT_EQ(inner.get_or_empty(kProbeName), "")
            << "set_fallback() overwrote a name that was present and set to "
               "the empty string";
        // get_or_empty() cannot tell absent from set-to-empty, so read the
        // distinction set_fallback() turns on through the seam itself.
        const std::optional<std::string> still_present =
            EnvironConfigBackend().get(kProbeName);
        ASSERT_TRUE(still_present.has_value())
            << "set_fallback() removed the name it was meant to leave alone";
        EXPECT_EQ(*still_present, "");
    }
}

// A skipped fallback write is not logged, so the destructor has nothing to
// replay for it and touches the backend not at all.
// Pseudo code:
//   probe = "set-deliberately"
//   { scope.set_fallback(probe, "fallback") }
//   assert no set or unset, and probe unchanged
TEST(ConfigScopeTest,
     set_fallback_with_an_already_present_value_then_logs_nothing) {
    FakeStore store;
    store.entries[kProbeName] = "set-deliberately";

    {
        ConfigScope config(std::make_unique<FakeBackend>(&store));
        config.set_fallback(kProbeName, "fallback");
    }

    EXPECT_EQ(store.set_count, 0);
    EXPECT_EQ(store.unset_count, 0)
        << "The scope restored a name that set_fallback() never wrote";
    EXPECT_EQ(store.entries[kProbeName], "set-deliberately");
}

// A fallback writes an absent name, and the scope undoes that write.
// Pseudo code:
//   assert probe absent
//   { scope.set_fallback(probe, "fallback"); assert probe == "fallback" }
//   assert probe absent
TEST(ConfigScopeTest,
     set_fallback_with_an_absent_name_then_writes_and_undoes_it) {
    ASSERT_EQ(getenv(kProbeName), nullptr)
        << "An earlier test in this suite did not restore the probe; leaked "
           "value: "
        << getenv(kProbeName);

    {
        ConfigScope config;
        config.set_fallback(kProbeName, "fallback");

        EXPECT_EQ(config.get_or_empty(kProbeName), "fallback")
            << "set_fallback() skipped a name that was absent";
    }

    EXPECT_EQ(getenv(kProbeName), nullptr)
        << "set_fallback() wrote without logging, so its write was not undone";
}

// get_or_empty()'s documented answer for a name that is not set.
// Pseudo code:
//   assert probe absent
//   assert scope.get_or_empty(probe) == ""
TEST(ConfigScopeTest,
     get_or_empty_with_an_absent_name_then_returns_the_empty_string) {
    ASSERT_EQ(getenv(kProbeName), nullptr)
        << "An earlier test in this suite did not restore the probe; leaked "
           "value: "
        << getenv(kProbeName);

    ConfigScope config;
    EXPECT_EQ(config.get_or_empty(kProbeName), "")
        << "get_or_empty() answered something other than the empty string "
           "for a name that is not set";
}
