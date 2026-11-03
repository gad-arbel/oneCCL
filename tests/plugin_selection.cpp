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

#include "internal/api/plugin.h"
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <dlfcn.h>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>

// Entrypoint of the null plugin, which the tests link
extern "C" void *onecclPluginCall(onecclPluginCall_t call_type);
static_assert(
    std::is_same_v<decltype(&onecclPluginCall), onecclPluginEntrypoint_t>);

namespace {

constexpr int kThreadCount = 64;
constexpr size_t kAllocSize = 64;
#ifdef ONECCL_TEST_GPU_PLUGIN_PATH
constexpr bool kHasGpuPlugin = true;
#else
constexpr bool kHasGpuPlugin = false;
#endif

// Set before any test calls oneCCL: the log level is read once, and
// ONEAPI_DEVICE_SELECTOR takes effect only before SYCL initializes
class PluginSelectionEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        // The first tests must see no plugin loaded
        if (GTEST_FLAG_GET(shuffle) || GTEST_FLAG_GET(repeat) != 1) {
            GTEST_FAIL() << "the tests depend on their order: run them "
                            "without --gtest_shuffle and --gtest_repeat";
        }
        unsetenv("CCL_PLUGIN");
        setenv("CCL_LOG_LEVEL", "info", 1);
        // Hide Level Zero from SYCL, as on a node without a GPU
        setenv("ONEAPI_DEVICE_SELECTOR", "opencl:*", 1);
    }
};

[[maybe_unused]] const auto *const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new PluginSelectionEnvironment);

bool library_loaded(const char *name) {
    void *handle = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        return false;
    }
    dlclose(handle);
    return true;
}

// Whether the loaded libccl.so.1 is the GPU build: only it has SYCL streams
bool gpu_libccl_v1_loaded() {
    void *handle = dlopen("libccl.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        return false;
    }
    bool const gpu =
        dlsym(handle, "_ZN3ccl6detail11environment13create_streamIN4sycl3_"
                      "V15queueEvEENS_2v16streamERT_") != nullptr;
    dlclose(handle);
    return gpu;
}

// Absolute path of the library that defines symbol, or empty
std::string library_path(void *symbol) {
    Dl_info info{};
    if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    return info.dli_fname;
}

// Path of the null plugin the test links. Look the entrypoint up in the
// library: the address the executable sees can be its own PLT entry.
std::string null_plugin_path() {
    void *const handle = dlopen("libccl_null.so", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        return {};
    }
    std::string path = library_path(dlsym(handle, "onecclPluginCall"));
    dlclose(handle);
    return path;
}

std::string last_error() { return onecclGetLastError(nullptr); }

// Tells the plugins apart: the null plugin implements onecclMemAlloc, the
// legacy CPU plugin returns onecclNotImplemented
bool is_cpu_plugin() {
    void *ptr = nullptr;
    onecclResult_t const result = onecclMemAlloc(&ptr, kAllocSize);
    if (result == onecclSuccess) {
        onecclMemFree(ptr);
    }
    return result == onecclNotImplemented;
}

// Runs body with CCL_PLUGIN set to value. Selection is per thread, so body
// runs in a new thread.
void run_with_override(const std::string &value,
                       const std::function<void()> &body) {
    setenv("CCL_PLUGIN", value.c_str(), 1);
    std::thread(body).join();
    unsetenv("CCL_PLUGIN");
}

void select_with_override(const std::string &value,
                          onecclResult_t *version_result, bool *cpu_plugin) {
    run_with_override(value, [=] {
        int version = -1;
        *version_result = onecclGetVersion(&version);
        *cpu_plugin = is_cpu_plugin();
    });
}

// Runs body with CCL_PLUGIN set to the fake plugin, behaving as mode says
void run_with_fake_plugin(const char *mode, const std::function<void()> &body) {
    setenv("ONECCL_TEST_FAKE_PLUGIN_MODE", mode, 1);
    run_with_override(ONECCL_TEST_FAKE_PLUGIN_PATH, body);
    unsetenv("ONECCL_TEST_FAKE_PLUGIN_MODE");
}

} // namespace

// Automatic selection only picks plugins with a positive score
//   score = null plugin PLATFORM_SCORE()
//   expect score <= 0
TEST(PluginSelection, NullPluginScoreIsNotPositive) {
    auto platform_score_func =
        reinterpret_cast<onecclPluginCallPlatformScore_t>(
            onecclPluginCall(ONECCL_PLUGIN_PLATFORM_SCORE));
    ASSERT_NE(platform_score_func, nullptr);

    // Output parameter. Starts positive, so the check fails if it is not set.
    int score = 1;
    ASSERT_EQ(platform_score_func(&score), onecclSuccess);
    EXPECT_LE(score, 0);
}

// A plugin the application loaded counts as a conflict, as it binds the
// GPU libccl.so.1. Runs before the CPU plugin is selected: the conflict check
// then skips the GPU plugin, not the CPU plugin.
//   application dlopens libccl_legacy.so
//   select in a new thread -> API fails, CPU plugin skipped as a conflict
//   application dlcloses it -> libccl_legacy.so and libccl.so.1 unloaded
TEST(PluginSelection, CpuPluginSkippedWhileApplicationHoldsGpuPlugin) {
    if (!kHasGpuPlugin) {
        GTEST_SKIP() << "no GPU plugin in this build";
    }
    void *const gpu_plugin = dlopen("libccl_legacy.so", RTLD_LAZY);
    ASSERT_NE(gpu_plugin, nullptr) << dlerror();

    onecclResult_t result = onecclSuccess;
    std::thread([&] {
        int version = -1;
        result = onecclGetVersion(&version);
    }).join();
    EXPECT_EQ(result, onecclError);
    EXPECT_NE(last_error().find("ONECCL_LEGACY_CPU: skipped"),
              std::string::npos)
        << last_error();

    // The next test loads the GPU plugin again and scores it
    dlclose(gpu_plugin);
    EXPECT_FALSE(library_loaded("libccl_legacy.so"));
    EXPECT_FALSE(library_loaded("libccl.so.1"));
}

// The GPU plugin loads first and fails its platform score, it must not keep
// the CPU plugin from loading. Every thread selects its plugin on its own.
// Runs before the CCL_PLUGIN tests: a CPU plugin they load would keep the GPU
// plugin from loading here.
//   no CCL_PLUGIN, Level Zero hidden
//   in kThreadCount threads at once and in the main thread:
//       onecclGetVersion() -> success
//       onecclMemAlloc()   -> not implemented (CPU plugin, not null plugin)
//   expect log: GPU plugin scored (if built), CPU plugin selected
//   expect libccl_legacy_cpu.so loaded, libccl_legacy.so not loaded
//   expect the CPU build of libccl.so.1 loaded, not the GPU build
TEST(PluginSelection, CpuPluginSelectedWithoutGpuInEveryThread) {
    testing::internal::CaptureStderr();

    std::atomic<bool> start{false};
    std::vector<onecclResult_t> results(kThreadCount, onecclError);
    // Not std::vector<bool>: its elements share bytes, so writes race
    std::array<bool, kThreadCount> cpu_plugin{};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; i++) {
        threads.emplace_back([&start, &results, &cpu_plugin, i] {
            while (!start) {
                std::this_thread::yield();
            }
            int version = -1;
            results[i] = onecclGetVersion(&version);
            cpu_plugin[i] = is_cpu_plugin();
        });
    }
    start = true;
    for (auto &thread : threads) {
        thread.join();
    }
    int version = -1;
    onecclResult_t const main_result = onecclGetVersion(&version);
    bool const main_cpu_plugin = is_cpu_plugin();

    std::string const log = testing::internal::GetCapturedStderr();

    for (int i = 0; i < kThreadCount; i++) {
        EXPECT_EQ(results[i], onecclSuccess) << "thread " << i;
        EXPECT_TRUE(cpu_plugin[i]) << "thread " << i;
    }
    EXPECT_EQ(main_result, onecclSuccess);
    EXPECT_TRUE(main_cpu_plugin);

    // The GPU plugin was loaded and scored, then the CPU plugin was selected
    if (kHasGpuPlugin) {
        EXPECT_NE(log.find("Platform score for ONECCL_LEGACY: "),
                  std::string::npos)
            << log;
    }
    EXPECT_NE(log.find("Selected plugin: ONECCL_LEGACY_CPU"), std::string::npos)
        << log;
    EXPECT_TRUE(library_loaded("libccl_legacy_cpu.so"));
    EXPECT_FALSE(library_loaded("libccl_legacy.so"));
    EXPECT_TRUE(library_loaded("libccl.so.1"));
    EXPECT_FALSE(gpu_libccl_v1_loaded());
}

// Threads select while threads of the previous round exit
//   kRounds times: kThreadCount threads select at once and exit
//   expect every thread: success, CPU plugin
//   expect libccl_legacy.so not loaded
TEST(PluginSelection, CpuPluginSelectedInRepeatedRounds) {
    constexpr int kRounds = 5;
    for (int round = 0; round < kRounds; round++) {
        std::atomic<bool> start{false};
        std::vector<onecclResult_t> results(kThreadCount, onecclError);
        std::array<bool, kThreadCount> cpu_plugin{};
        std::vector<std::thread> threads;
        threads.reserve(kThreadCount);
        for (int i = 0; i < kThreadCount; i++) {
            threads.emplace_back([&start, &results, &cpu_plugin, i] {
                while (!start) {
                    std::this_thread::yield();
                }
                int version = -1;
                results[i] = onecclGetVersion(&version);
                cpu_plugin[i] = is_cpu_plugin();
            });
        }
        start = true;
        for (auto &thread : threads) {
            thread.join();
        }
        for (int i = 0; i < kThreadCount; i++) {
            EXPECT_EQ(results[i], onecclSuccess)
                << "round " << round << " thread " << i;
            EXPECT_TRUE(cpu_plugin[i]) << "round " << round << " thread " << i;
        }
    }
    EXPECT_FALSE(library_loaded("libccl_legacy.so"));
}

// A thread whose selection failed selects again on its next call
//   CCL_PLUGIN=ONECCL_NO_SUCH_PLUGIN, thread calls the API -> fails
//   CCL_PLUGIN unset, same thread calls again -> success, CPU plugin
TEST(PluginSelection, FailedThreadSelectsAgain) {
    std::atomic<bool> first_done{false};
    std::atomic<bool> env_reset{false};
    onecclResult_t first_result = onecclSuccess;
    onecclResult_t retry_result = onecclError;
    bool cpu_plugin = false;

    setenv("CCL_PLUGIN", "ONECCL_NO_SUCH_PLUGIN", 1);
    std::thread thread([&] {
        int version = -1;
        first_result = onecclGetVersion(&version);
        first_done = true;
        while (!env_reset) {
            std::this_thread::yield();
        }
        retry_result = onecclGetVersion(&version);
        cpu_plugin = is_cpu_plugin();
    });
    while (!first_done) {
        std::this_thread::yield();
    }
    unsetenv("CCL_PLUGIN");
    env_reset = true;
    thread.join();

    EXPECT_EQ(first_result, onecclError);
    EXPECT_EQ(retry_result, onecclSuccess);
    EXPECT_TRUE(cpu_plugin);
}

// Selections of other threads do not change the plugin of a thread
//   thread A: CCL_PLUGIN=ONECCL_NULL -> null plugin, keeps calling it
//   CCL_PLUGIN unset, 16 threads select at once -> CPU plugin
//   expect thread A on the null plugin throughout
TEST(PluginSelection, SelectedPluginKeptWhileOtherThreadsSelect) {
    constexpr int kOtherThreads = 16;
    std::atomic<bool> selected{false};
    std::atomic<bool> env_reset{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> left_null_plugin{false};
    onecclResult_t override_result = onecclError;

    setenv("CCL_PLUGIN", "ONECCL_NULL", 1);
    std::thread override_thread([&] {
        int version = -1;
        override_result = onecclGetVersion(&version);
        selected = true;
        while (!env_reset) {
            std::this_thread::yield();
        }
        while (!stop) {
            if (is_cpu_plugin()) {
                left_null_plugin = true;
            }
        }
    });
    while (!selected) {
        std::this_thread::yield();
    }
    unsetenv("CCL_PLUGIN");
    env_reset = true;

    std::vector<onecclResult_t> results(kOtherThreads, onecclError);
    std::array<bool, kOtherThreads> cpu_plugin{};
    std::vector<std::thread> threads;
    threads.reserve(kOtherThreads);
    for (int i = 0; i < kOtherThreads; i++) {
        threads.emplace_back([&results, &cpu_plugin, i] {
            int version = -1;
            results[i] = onecclGetVersion(&version);
            cpu_plugin[i] = is_cpu_plugin();
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    stop = true;
    override_thread.join();

    EXPECT_EQ(override_result, onecclSuccess);
    EXPECT_FALSE(left_null_plugin);
    for (int i = 0; i < kOtherThreads; i++) {
        EXPECT_EQ(results[i], onecclSuccess) << "thread " << i;
        EXPECT_TRUE(cpu_plugin[i]) << "thread " << i;
    }
}

// A forked child keeps the selection, and its new threads select again
//   main thread selected the CPU plugin (earlier test)
//   fork; in the child: main thread and a new thread -> CPU plugin
TEST(PluginSelection, ForkedChildKeepsSelection) {
    int version = -1;
    ASSERT_EQ(onecclGetVersion(&version), onecclSuccess);
    ASSERT_TRUE(is_cpu_plugin());

    pid_t const pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
        bool ok =
            onecclGetVersion(&version) == onecclSuccess && is_cpu_plugin();
        std::thread([&] {
            int child_version = -1;
            ok = ok && onecclGetVersion(&child_version) == onecclSuccess &&
                 is_cpu_plugin();
        }).join();
        _exit(ok ? 0 : 1);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << status;
}

//   CCL_PLUGIN=ONECCL_NULL       -> onecclMemAlloc() succeeds (null)
//   CCL_PLUGIN=ONECCL_LEGACY_CPU -> onecclMemAlloc() not implemented (CPU)
TEST(PluginSelection, OverrideSelectsPluginByName) {
    onecclResult_t version_result = onecclError;
    bool cpu_plugin = false;

    select_with_override("ONECCL_NULL", &version_result, &cpu_plugin);
    EXPECT_EQ(version_result, onecclSuccess);
    EXPECT_FALSE(cpu_plugin);

    select_with_override("ONECCL_LEGACY_CPU", &version_result, &cpu_plugin);
    EXPECT_EQ(version_result, onecclSuccess);
    EXPECT_TRUE(cpu_plugin);
}

//   path = absolute path of the linked libccl_null.so
//   CCL_PLUGIN=path -> onecclMemAlloc() succeeds (null plugin)
TEST(PluginSelection, OverrideSelectsPluginByPath) {
    std::string const path = null_plugin_path();
    ASSERT_FALSE(path.empty());
    ASSERT_EQ(path.front(), '/') << path;

    onecclResult_t version_result = onecclError;
    bool cpu_plugin = false;
    select_with_override(path, &version_result, &cpu_plugin);
    EXPECT_EQ(version_result, onecclSuccess);
    EXPECT_FALSE(cpu_plugin);
}

// No plugin is selected: the API fails and reports why
//   CCL_PLUGIN=ONECCL_NO_SUCH_PLUGIN     -> API fails, error names CCL_PLUGIN
//   CCL_PLUGIN=/no/such/libccl_plugin.so -> API fails, error names the path
TEST(PluginSelection, OverrideReportsPluginThatCannotLoad) {
    onecclResult_t version_result = onecclSuccess;
    onecclResult_t alloc_result = onecclSuccess;
    auto const call_api = [&] {
        int version = -1;
        version_result = onecclGetVersion(&version);
        void *ptr = nullptr;
        alloc_result = onecclMemAlloc(&ptr, kAllocSize);
    };

    run_with_override("ONECCL_NO_SUCH_PLUGIN", call_api);
    EXPECT_EQ(version_result, onecclError);
    EXPECT_EQ(alloc_result, onecclError);
    EXPECT_NE(last_error().find("CCL_PLUGIN"), std::string::npos)
        << last_error();

    run_with_override("/no/such/libccl_plugin.so", call_api);
    EXPECT_EQ(version_result, onecclError);
    EXPECT_EQ(alloc_result, onecclError);
    EXPECT_NE(std::string(onecclGetLastError(nullptr))
                  .find("dlopen(/no/such/libccl_plugin.so)"),
              std::string::npos)
        << onecclGetLastError(nullptr);
}

// The thread selected its plugin, a later CCL_PLUGIN does not change it
//   main thread selected the CPU plugin (earlier test)
//   CCL_PLUGIN=ONECCL_NULL, same thread calls again -> CPU plugin
TEST(PluginSelection, OverrideIgnoredAfterThreadSelected) {
    int version = -1;
    ASSERT_EQ(onecclGetVersion(&version), onecclSuccess);
    ASSERT_TRUE(is_cpu_plugin());

    setenv("CCL_PLUGIN", "ONECCL_NULL", 1);
    EXPECT_EQ(onecclGetVersion(&version), onecclSuccess);
    EXPECT_TRUE(is_cpu_plugin());
    unsetenv("CCL_PLUGIN");
}

//   CCL_PLUGIN="" -> automatic selection, CPU plugin
TEST(PluginSelection, OverrideEmptyValueSelectsAutomatically) {
    onecclResult_t version_result = onecclError;
    bool cpu_plugin = false;
    select_with_override("", &version_result, &cpu_plugin);
    EXPECT_EQ(version_result, onecclSuccess);
    EXPECT_TRUE(cpu_plugin);
}

// Names match exactly, and a path must be absolute and free of ".."
//   CCL_PLUGIN=<each value> -> API fails
TEST(PluginSelection, OverrideRejectsMalformedValue) {
    std::string const path = null_plugin_path();
    ASSERT_FALSE(path.empty());
    // The same existing file, reached through ".."
    std::string const dir = path.substr(0, path.rfind('/'));
    std::string const dotdot_path =
        dir + "/../" + dir.substr(dir.rfind('/') + 1) + "/libccl_null.so";
    ASSERT_EQ(access(dotdot_path.c_str(), F_OK), 0) << dotdot_path;

    for (const std::string &value :
         {std::string(" "), std::string("oneccl_null"),
          std::string(" ONECCL_NULL"), std::string("ONECCL_NULL "),
          std::string("ONECCL_PLUGIN_ANY"), std::string("libccl_null.so"),
          std::string("./libccl_null.so"), dotdot_path}) {
        onecclResult_t version_result = onecclSuccess;
        run_with_override(value, [&] {
            int version = -1;
            version_result = onecclGetVersion(&version);
        });
        EXPECT_EQ(version_result, onecclError) << "[" << value << "]";
    }
}

// A file that is not a plugin fails, and the error says why
//   CCL_PLUGIN=<libc path>             -> API fails, error names dlsym
//   CCL_PLUGIN=<plugin directory path> -> API fails, error names dlopen
TEST(PluginSelection, OverrideRejectsFileThatIsNotPlugin) {
    // Look malloc up in libc: the executable can define its own
    void *const libc = dlopen("libc.so.6", RTLD_LAZY | RTLD_NOLOAD);
    ASSERT_NE(libc, nullptr);
    std::string const libc_path = library_path(dlsym(libc, "malloc"));
    dlclose(libc);
    ASSERT_FALSE(libc_path.empty());
    ASSERT_EQ(libc_path.front(), '/') << libc_path;
    std::string const path = null_plugin_path();
    ASSERT_FALSE(path.empty());
    std::string const dir = path.substr(0, path.rfind('/') + 1);

    onecclResult_t version_result = onecclSuccess;
    auto const call_api = [&] {
        int version = -1;
        version_result = onecclGetVersion(&version);
    };

    run_with_override(libc_path, call_api);
    EXPECT_EQ(version_result, onecclError);
    EXPECT_NE(last_error().find("dlsym("), std::string::npos) << last_error();

    run_with_override(dir, call_api);
    EXPECT_EQ(version_result, onecclError);
    EXPECT_NE(last_error().find("dlopen("), std::string::npos) << last_error();
}

// A plugin whose init fails or throws is not selected and is unloaded
//   CCL_PLUGIN=<fake plugin>, init returns an error -> API fails
//   CCL_PLUGIN=<fake plugin>, init throws            -> API fails
//   expect the fake plugin not loaded afterwards
TEST(PluginSelection, OverrideReportsPluginInitFailure) {
    for (const char *mode : {"init_fail", "init_throw"}) {
        onecclResult_t version_result = onecclSuccess;
        run_with_fake_plugin(mode, [&] {
            int version = -1;
            version_result = onecclGetVersion(&version);
        });
        EXPECT_EQ(version_result, onecclError) << mode;
        EXPECT_NE(last_error().find("plugin init failed"), std::string::npos)
            << mode << ": " << last_error();
        EXPECT_FALSE(library_loaded(ONECCL_TEST_FAKE_PLUGIN_PATH)) << mode;
    }
}

// An exception from the plugin entrypoint itself must not leak the library
//   CCL_PLUGIN=<fake plugin>, onecclPluginCall(INIT) throws -> API fails
//   expect the fake plugin not loaded afterwards
TEST(PluginSelection, OverrideReleasesPluginWhoseEntrypointThrows) {
    onecclResult_t version_result = onecclSuccess;
    run_with_fake_plugin("entrypoint_throw", [&] {
        int version = -1;
        version_result = onecclGetVersion(&version);
    });
    EXPECT_EQ(version_result, onecclError);
    EXPECT_FALSE(library_loaded(ONECCL_TEST_FAKE_PLUGIN_PATH));
}

// fork() waits for a selection in progress, so the child can select
//   thread A: CCL_PLUGIN=<fake plugin>, its init sleeps
//   main thread forks while A selects
//   in the child: a new thread selects -> success, no deadlock
// Leaves the fake plugin loaded
TEST(PluginSelection, ChildForkedDuringSelectionCanSelect) {
    constexpr auto kForkDelay = std::chrono::milliseconds(100);
    constexpr unsigned kChildTimeoutSeconds = 10;

    setenv("ONECCL_TEST_FAKE_PLUGIN_MODE", "init_sleep", 1);
    setenv("CCL_PLUGIN", ONECCL_TEST_FAKE_PLUGIN_PATH, 1);
    onecclResult_t selecting_result = onecclError;
    std::thread selecting_thread([&] {
        int version = -1;
        selecting_result = onecclGetVersion(&version);
    });
    std::this_thread::sleep_for(kForkDelay);

    pid_t const pid = fork();
    if (pid == 0) {
        alarm(kChildTimeoutSeconds);
        unsetenv("CCL_PLUGIN");
        onecclResult_t child_result = onecclError;
        std::thread([&] {
            int version = -1;
            child_result = onecclGetVersion(&version);
        }).join();
        _exit(child_result == onecclSuccess ? 0 : 1);
    }
    selecting_thread.join();
    unsetenv("CCL_PLUGIN");
    unsetenv("ONECCL_TEST_FAKE_PLUGIN_MODE");

    ASSERT_NE(pid, -1);
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    EXPECT_FALSE(WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM)
        << "the child deadlocked";
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << status;
    EXPECT_EQ(selecting_result, onecclSuccess);
}

// A plugin that calls the API from its init gets an error, not a deadlock
//   CCL_PLUGIN=<fake plugin>, init calls onecclGetVersion()
//   expect the inner call fails, the outer call selects the fake plugin
// Leaves the fake plugin loaded
TEST(PluginSelection, PluginCallingApiDuringSelectionGetsError) {
    onecclResult_t version_result = onecclError;
    onecclResult_t alloc_result = onecclError;
    run_with_fake_plugin("init_reenter", [&] {
        int version = -1;
        version_result = onecclGetVersion(&version);
        void *ptr = nullptr;
        alloc_result = onecclMemAlloc(&ptr, kAllocSize);
    });
    EXPECT_EQ(version_result, onecclSuccess);
    EXPECT_EQ(alloc_result, onecclSuccess); // The fake plugin, selected

    void *const handle =
        dlopen(ONECCL_TEST_FAKE_PLUGIN_PATH, RTLD_LAZY | RTLD_NOLOAD);
    ASSERT_NE(handle, nullptr);
    const auto *const reentry_result = static_cast<const onecclResult_t *>(
        dlsym(handle, "fake_plugin_reentry_result"));
    dlclose(handle);
    ASSERT_NE(reentry_result, nullptr);
    EXPECT_EQ(*reentry_result, onecclError);
}
