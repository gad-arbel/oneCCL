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

#define CCL_C_API_EXPORT

#include "internal/api/comm.h"
#include "internal/api/plugin.h"
#include "internal/debug.h"
#include "internal/dlopen_compat.h"
#include "internal/utils_compat.h"
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <pthread.h>
#endif

namespace {
thread_local struct onecclThreadContext {
    bool init_done = false;
    onecclPluginType_t plugin_type;
    onecclPlugin_t plugin;
} thread_context;

// Serializes plugin loading and unloading across threads, so a plugin that is
// released by one init attempt is not unloaded under another one, and the
// conflict checks see a stable set of loaded libraries
std::mutex plugin_load_mutex;
// Number of dlopen() references held on each plugin type, by the plugins
// selected by all threads and by the init attempt in progress. Guarded by
// plugin_load_mutex.
std::map<onecclPluginType_t, int> plugin_ref_counts;
// Caller holds plugin_load_mutex
void plugin_refcount_inc(onecclPluginType_t type) { plugin_ref_counts[type]++; }
// Caller holds plugin_load_mutex
void plugin_refcount_dec(onecclPluginType_t type) noexcept {
    auto const refs = plugin_ref_counts.find(type);
    if (refs != plugin_ref_counts.end()) {
        refs->second--;
    }
}
// Why plugins could not be used in the current init attempt, reported if no
// plugin is selected
thread_local std::string plugin_load_errors;
// Whether libccl.so.1 was loaded before the first init attempt (e.g. linked
// by the application). Taken once, so a copy that a released plugin left
// behind does not count as preloaded in later attempts. Guarded by
// plugin_load_mutex.
std::optional<bool> libccl_v1_preloaded;

// Plugins the current init attempt opened and then released, which may have
// left their libccl.so.1 loaded
thread_local std::vector<onecclPluginType_t> plugins_with_stale_libccl;

bool may_have_stale_libccl(onecclPluginType_t type) {
    return std::find(plugins_with_stale_libccl.begin(),
                     plugins_with_stale_libccl.end(),
                     type) != plugins_with_stale_libccl.end();
}
// Set while the thread selects its plugin, to catch plugin code that calls
// back into the API on the same thread. A call from another thread that the
// plugin waits for still deadlocks.
thread_local bool selecting_plugin = false;

#ifndef _WIN32
// fork() waits for a selection in progress: a child forked during one would
// find plugin_load_mutex held by a thread it does not have. The selecting
// thread itself already holds it.
void lock_plugin_selection_for_fork() {
    if (!selecting_plugin) {
        plugin_load_mutex.lock();
    }
}

void unlock_plugin_selection_after_fork() {
    if (!selecting_plugin) {
        plugin_load_mutex.unlock();
    }
}

[[maybe_unused]] const int kForkHandlers = pthread_atfork(
    lock_plugin_selection_for_fork, unlock_plugin_selection_after_fork,
    unlock_plugin_selection_after_fork);
#endif

// This a map is storing possible names for plugins used by oneCCL,
// The name `libexternal_Soneccl_Slibccl_Ulegacy` was added to support
// XLA and it's build system based on bazel.
const std::map<onecclPluginType_t, std::vector<std::string>> kPluginPaths = {
    {onecclLegacy,
     {"libccl_legacy.so", "libexternal_Soneccl_Slibccl_Ulegacy.so"}},
    {onecclLegacyCPU, {"libccl_legacy_cpu.so"}},
    {onecclNull, {"libccl_null.so"}}};

// Safe call which will never throw exceptions
template <typename Func, typename... Args>
onecclResult_t __safe_call_impl(const char *func_name, Func &&func,
                                Args &&...args) noexcept {
    try {
        return func(std::forward<Args>(args)...);
    } catch (const std::exception &ex) {
        ONECCL_ERROR("[%s] caught std::exception: %s", func_name, ex.what());
        return onecclPluginException;
    } catch (...) {
        ONECCL_ERROR("[%s] caught unknown exception", func_name);
        return onecclPluginException;
    }
}

#define SAFE_CALL(X, ...) __safe_call_impl(__func__, X, ##__VA_ARGS__)

#define ONECCL_VARNAME(x) #x

#define ONECCL_CHECK_PTR(x)                                                    \
    if ((x) == nullptr) {                                                      \
        ONECCL_WARN(ONECCL_VARNAME(x) " cannot be nullptr");                   \
        return onecclInvalidArgument;                                          \
    }

#define ONECCL_CHECK_MALLOC(x)                                                 \
    if ((x) == nullptr) {                                                      \
        ONECCL_ERROR("failed to allocate memory for " ONECCL_VARNAME(x));      \
        return onecclAllocFailureCPU;                                          \
    }

#ifndef BUILD_GIT_COMMIT
#define BUILD_GIT_COMMIT "unknown"
#endif
#ifndef BUILD_TIME_UTC
#define BUILD_TIME_UTC "unknown"
#endif
#ifndef BUILD_COMPILER_ID
#define BUILD_COMPILER_ID "unknown"
#endif
#ifndef BUILD_COMPILER_VERSION
#define BUILD_COMPILER_VERSION "unknown"
#endif
#ifndef BUILD_COMPILER_PATH
#define BUILD_COMPILER_PATH "unknown"
#endif
#ifndef BUILD_TYPE
#define BUILD_TYPE "unknown"
#endif
#ifndef BUILD_SYSTEM_NAME
#define BUILD_SYSTEM_NAME "unknown"
#endif
#ifndef BUILD_SYSTEM_PROCESSOR
#define BUILD_SYSTEM_PROCESSOR "unknown"
#endif

std::string to_string(onecclPluginType_t type) {
    switch (type) {
    case onecclPluginAny:
        return "ONECCL_PLUGIN_ANY";
    case onecclLegacy:
        return "ONECCL_LEGACY";
    case onecclLegacyCPU:
        return "ONECCL_LEGACY_CPU";
    case onecclNull:
        return "ONECCL_NULL";
    default:
        return "UNKNOWN";
    }
}

struct onecclLoadedPlugin;
void release_plugin(onecclLoadedPlugin *loaded) noexcept;

// A plugin loaded by one init attempt. It owns one dlopen() reference to the
// library and gives it back when destroyed, also when an exception unwinds
// the attempt. select_plugin() moves the reference to the selecting thread,
// which keeps it for the lifetime of the process. Move-only, so the reference
// has one owner. Destroy only while holding plugin_load_mutex.
struct onecclLoadedPlugin {
    onecclPlugin_t plugin{};
    dl_handle handle = nullptr;
    onecclPluginType_t type = onecclPluginAny;

    onecclLoadedPlugin() = default;
    onecclLoadedPlugin(const onecclLoadedPlugin &) = delete;
    onecclLoadedPlugin &operator=(const onecclLoadedPlugin &) = delete;
    onecclLoadedPlugin(onecclLoadedPlugin &&other) noexcept {
        *this = std::move(other);
    }
    onecclLoadedPlugin &operator=(onecclLoadedPlugin &&other) noexcept {
        if (this != &other) {
            release_plugin(this);
            plugin = other.plugin;
            handle = std::exchange(other.handle, nullptr);
            type = other.type;
        }
        return *this;
    }
    ~onecclLoadedPlugin() { release_plugin(this); }
};

// Drop the reference taken by load_plugin(); the loader unloads the library
// (and its dependencies such as libccl.so.1) once no thread holds it.
// Caller holds plugin_load_mutex.
void release_plugin(onecclLoadedPlugin *loaded) noexcept {
    if (loaded->handle == nullptr) {
        return;
    }
    plugin_refcount_dec(loaded->type);
    dlclose(std::exchange(loaded->handle, nullptr));
    // Cannot throw: init_library() reserves room for every plugin type
    if (!may_have_stale_libccl(loaded->type)) {
        plugins_with_stale_libccl.push_back(loaded->type);
    }
    loaded->plugin = {};
    loaded->type = onecclPluginAny;
}

// Caller holds plugin_load_mutex
onecclResult_t load_plugin(onecclLoadedPlugin *loaded, const std::string &path,
                           onecclPluginType_t type) {
    if (loaded == nullptr) {
        return onecclInvalidArgument;
    }

    // Each call takes its own reference, so releasing it cannot unload a
    // library another thread still uses
    dl_handle handle = dlopen(path.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        const char *error = dlerror();
        ONECCL_INFO("Cannot open plugin library: %s", error);
        plugin_load_errors += "\n  dlopen(" + path + "): ";
        plugin_load_errors += error != nullptr ? error : "unknown error";
        return onecclError;
    }
    ONECCL_INFO("Loaded plugin library: %s", path.c_str());
    plugin_refcount_inc(type);
    loaded->handle = handle;
    loaded->type = type;

    dlerror(); // Clear any existing error

    // Get pointer the plugin entrypoint
    const char *symbol = "onecclPluginCall";
    auto plugin_call =
        reinterpret_cast<onecclPluginEntrypoint_t>(dlsym(handle, symbol));
    const char *dlsym_error = dlerror();
    if (dlsym_error != nullptr) {
        ONECCL_ERROR("Cannot load plugin entrypoint: '%s': %s", symbol,
                     dlsym_error);
        plugin_load_errors +=
            "\n  dlsym(" + path + ", " + symbol + "): " + dlsym_error;
        release_plugin(loaded);
        return onecclError;
    }

    loaded->plugin.call = plugin_call;

    // The entrypoint lookup runs plugin code too, so it can throw
    onecclResult_t const result = SAFE_CALL([&] {
        auto plugin_init_fn = reinterpret_cast<onecclPluginCallInit_t>(
            loaded->plugin.call(ONECCL_PLUGIN_INIT));
        return plugin_init_fn != nullptr ? plugin_init_fn() : onecclSuccess;
    });
    if (result != onecclSuccess) {
        plugin_load_errors += "\n  plugin init failed: " + path;
        release_plugin(loaded);
    }
    return result;
}

// Directory of libccl.so.2 itself (with a trailing slash), or empty
std::string get_library_dir() {
#ifndef _WIN32
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&get_library_dir), &info) != 0 &&
        info.dli_fname != nullptr) {
        std::string const path(info.dli_fname);
        auto const pos = path.rfind('/');
        if (pos != std::string::npos) {
            return path.substr(0, pos + 1);
        }
    }
#endif
    return {};
}

// Whether a library of the name is loaded in the process
bool library_is_loaded_in_process(const char *name) {
#ifdef _WIN32
    return false; // Cannot tell
#else
    dl_handle handle = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) {
        return false;
    }
    dlclose(handle); // Drop the reference RTLD_NOLOAD took
    return true;
#endif
}

// Prefer the plugin installed next to libccl.so.2, so the plugin matches this
// libccl.so.2 no matter how it was found; if it is missing or dlopen fails,
// use the loader search. A plugin the process already loaded (e.g. linked by
// the application) wins, so no second copy with separate state is opened.
onecclResult_t load_bundled_plugin(onecclLoadedPlugin *loaded,
                                   const std::string &name,
                                   onecclPluginType_t type) {
#ifndef _WIN32
    if (library_is_loaded_in_process(name.c_str())) {
        return load_plugin(loaded, name, type);
    }
    static const std::string library_dir = get_library_dir();
    std::string const sibling = library_dir + name;
    if (!library_dir.empty() && access(sibling.c_str(), F_OK) == 0) {
        if (load_plugin(loaded, sibling, type) == onecclSuccess) {
            return onecclSuccess;
        }
        onecclResult_t const result = load_plugin(loaded, name, type);
        if (result == onecclSuccess) {
            ONECCL_WARN("Could not use plugin %s; loaded %s from the library "
                        "search path instead, which may be a different oneCCL "
                        "build",
                        sibling.c_str(), name.c_str());
        }
        return result;
    }
#endif
    return load_plugin(loaded, name, type);
}

// Make the plugin the one used by the calling thread. The thread keeps the
// reference to the library for the lifetime of the process.
void select_plugin(onecclLoadedPlugin &&loaded) {
    ONECCL_INFO("Selected plugin: %s", to_string(loaded.type).c_str());
    thread_context.plugin = loaded.plugin;
    thread_context.plugin_type = loaded.type;
    thread_context.init_done = true;
    loaded.handle = nullptr; // Kept by the thread, not released
}

// Whether a library of the plugin is loaded in the process
bool plugin_is_loaded_in_process(onecclPluginType_t type) {
    for (const auto &path : kPluginPaths.at(type)) {
        if (library_is_loaded_in_process(path.c_str())) {
            return true;
        }
    }
    return false;
}

// Whether a plugin of the type cannot be loaded now: a plugin that needs
// libccl.so.1 by the same soname is loaded, so this one would bind to that
// one's copy. Caller holds plugin_load_mutex.
bool plugin_conflicts(onecclPluginType_t type) {
    static const std::map<onecclPluginType_t, onecclPluginType_t> kConflicts = {
        {onecclLegacy, onecclLegacyCPU}, {onecclLegacyCPU, onecclLegacy}};
    auto const conflict = kConflicts.find(type);
    if (conflict == kConflicts.end()) {
        return false;
    }
    onecclPluginType_t const other = conflict->second;
    // Loaded by this attempt, selected by a thread, or loaded by the
    // application and so kept after its release
    auto const refs = plugin_ref_counts.find(other);
    if ((refs != plugin_ref_counts.end() && refs->second > 0) ||
        plugin_is_loaded_in_process(other)) {
        return true;
    }
    // Released by this attempt, but its libccl.so.1 stayed
    return may_have_stale_libccl(other) &&
           !libccl_v1_preloaded.value_or(false) &&
           library_is_loaded_in_process("libccl.so.1");
}

onecclResult_t init_library_with_override(const char *ccl_plugin) {
    ONECCL_INFO("Using CCL_PLUGIN override (%s)", ccl_plugin);
    std::string plugin_str(ccl_plugin);

    // Check if it matches any known plugin names
    for (const auto &[type, paths] : kPluginPaths) {
        if (plugin_str == to_string(type)) {
            for (const auto &path : paths) {
                onecclLoadedPlugin loaded;
                onecclResult_t const result =
                    load_bundled_plugin(&loaded, path, type);
                if (result == onecclSuccess) {
                    select_plugin(std::move(loaded));
                    return onecclSuccess;
                }
            }
            break; // If loading fails, proceed to the default procedure
        }
    }

    // Check if it's a direct file path
    if (plugin_str.front() == '/' &&
        plugin_str.find("..") == std::string::npos) {
        onecclLoadedPlugin loaded;
        onecclResult_t const result =
            load_plugin(&loaded, plugin_str, onecclUserBackend);
        if (result == onecclSuccess) {
            select_plugin(std::move(loaded));
            return onecclSuccess;
        }
    }

    ONECCL_ERROR("oneCCL could not initialize plugin provided in CCL_PLUGIN.%s",
                 plugin_load_errors.c_str());
    return onecclError;
}

// Platform score of the plugin, or none if it cannot give one. A plugin that
// throws here, e.g. as it finds no device, is only not selected: logged at
// info level, not as an error.
std::optional<int> platform_score(const onecclLoadedPlugin &candidate) {
    std::string const name = to_string(candidate.type);
    try {
        auto platform_score_func =
            reinterpret_cast<onecclPluginCallPlatformScore_t>(
                candidate.plugin.call(ONECCL_PLUGIN_PLATFORM_SCORE));
        if (platform_score_func == nullptr) {
            ONECCL_ERROR(
                "Plugin does not provide function to evaluate platform!");
            plugin_load_errors +=
                "\n  " + name + ": no platform evaluation function";
            return std::nullopt;
        }
        int score = INT_MIN;
        if (platform_score_func(&score) != onecclSuccess) {
            ONECCL_INFO("Failed to evaluate plugin score!");
            plugin_load_errors +=
                "\n  " + name + ": platform evaluation failed";
            return std::nullopt;
        }
        return score;
    } catch (const std::exception &ex) {
        ONECCL_INFO("Platform evaluation of %s threw: %s", name.c_str(),
                    ex.what());
        plugin_load_errors +=
            "\n  " + name + ": platform evaluation threw: " + ex.what();
    } catch (...) {
        ONECCL_INFO("Platform evaluation of %s threw", name.c_str());
        plugin_load_errors += "\n  " + name + ": platform evaluation threw";
    }
    return std::nullopt;
}

onecclResult_t init_library_without_override() {
    ONECCL_INFO("Proceeding without CCL_PLUGIN override.");
    int best_platform_score = INT_MIN;
    onecclLoadedPlugin best;

    for (const auto &[type, paths] : kPluginPaths) {
        if (plugin_conflicts(type)) {
            // We already loaded a plugin that is conflicting with plugin of
            // `type`, so we cannot load this one now.
            ONECCL_INFO("Failed to load plugin type: %s due to conflicts "
                        "with previously loaded plugins.",
                        to_string(type).c_str());
            plugin_load_errors += "\n  " + to_string(type) +
                                  ": skipped, conflicts with a loaded plugin";
            continue;
        }

        onecclLoadedPlugin candidate;
        onecclResult_t result = onecclError;
        for (const auto &path : paths) {
            result = load_bundled_plugin(&candidate, path, type);
            if (result == onecclSuccess) {
                break; // Successfully loaded, no need to try other paths
            }
        }
        if (result != onecclSuccess) {
            ONECCL_INFO("Failed to load plugin, type: %s",
                        to_string(type).c_str());
            continue;
        }

        std::optional<int> const score = platform_score(candidate);
        if (score.has_value()) {
            ONECCL_INFO("Platform score for %s: %d", to_string(type).c_str(),
                        *score);
            if (*score <= 0) {
                plugin_load_errors += "\n  " + to_string(type) +
                                      ": platform score " +
                                      std::to_string(*score);
            }
        }

        if (score.has_value() && *score > 0 && *score > best_platform_score) {
            best = std::move(candidate); // Releases the previous best
            best_platform_score = *score;
        } else {
            // Not used: unload it and its libccl.so.1, so a conflicting
            // plugin can still load. The SYCL runtime that scoring brought
            // up stays loaded: it is marked NODELETE.
            release_plugin(&candidate);
        }
    }

    if (best.handle == nullptr) {
        ONECCL_ERROR("oneCCL could not be initialized! Could not load any "
                     "plugin. Please rerun with `CCL_LOG_LEVEL=info`.%s",
                     plugin_load_errors.c_str());
        return onecclError;
    }

    select_plugin(std::move(best));
    return onecclSuccess;
}

onecclResult_t init_library() {
    onecclDebugInit();
    if (thread_context.init_done) {
        return onecclSuccess;
    }
    // Waiting for plugin_load_mutex would deadlock: this thread holds it
    if (selecting_plugin) {
        ONECCL_ERROR("oneCCL API called by a plugin during plugin selection");
        return onecclError;
    }

    ONECCL_INFO("\n  oneCCL V2 build info:\n"
                "    ONECCL_VERSION      : %d.%d.%d\n"
                "    GIT_COMMIT          : %s\n"
                "    BUILD_TIME_UTC      : %s\n"
                "    COMPILER_ID         : %s\n"
                "    COMPILER_VERSION    : %s\n"
                "    COMPILER_PATH       : %s\n"
                "    BUILD_TYPE          : %s\n"
                "    SYSTEM_NAME         : %s\n"
                "    SYSTEM_PROCESSOR    : %s\n",
                ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH, BUILD_GIT_COMMIT,
                BUILD_TIME_UTC, BUILD_COMPILER_ID, BUILD_COMPILER_VERSION,
                BUILD_COMPILER_PATH, BUILD_TYPE, BUILD_SYSTEM_NAME,
                BUILD_SYSTEM_PROCESSOR);

    plugin_load_errors.clear();
    plugins_with_stale_libccl.clear();
    // release_plugin() records each type at most once, and must not allocate
    plugins_with_stale_libccl.reserve(kPluginPaths.size() + 1);

    // Each thread selects its own plugin, one thread at a time
    std::scoped_lock const lock(plugin_load_mutex);
    selecting_plugin = true;
    if (!libccl_v1_preloaded.has_value()) {
        libccl_v1_preloaded = library_is_loaded_in_process("libccl.so.1");
    }

    // Try initializing with CCL_PLUGIN environment variable if set
    // No exception may skip the reset of selecting_plugin
    onecclResult_t const result = SAFE_CALL([] {
        const char *ccl2_plugin = utils_getenv("CCL_PLUGIN");
        return ccl2_plugin == nullptr || *ccl2_plugin == '\0'
                   ? init_library_without_override()
                   : init_library_with_override(ccl2_plugin);
    });

    selecting_plugin = false;
    return result;
}

#define INIT_CCL                                                               \
    if (init_library() != onecclSuccess)                                       \
    return onecclError

onecclResult_t init_comm(onecclComm_t *comm, size_t nranks,
                         const onecclUniqueId &commId, int rank,
                         const onecclConfig_t *config) {
    auto init_comm_func = reinterpret_cast<onecclPluginCallCommInit_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_COMM));
    if (init_comm_func == nullptr) {
        return onecclNotImplemented;
    }

    if (comm == nullptr || config == nullptr) {
        return onecclInvalidArgument;
    }

    auto *new_comm =
        static_cast<onecclComm_t>(calloc(1, sizeof(struct onecclComm)));
    ONECCL_CHECK_MALLOC(new_comm);

    new_comm->id = commId;
    new_comm->config = *config;

    auto result = SAFE_CALL([&] {
        return init_comm_func(&new_comm, nranks, commId, rank, config);
    });
    if (result != onecclSuccess) {
        ONECCL_ERROR("Plugin could not initialize communicator!");
        free(new_comm);
        return result;
    }

    *comm = new_comm;
    return onecclSuccess;
}

} // namespace

onecclResult_t onecclGetVersion(int *version) {
    INIT_CCL;

    *version = ONECCL_VERSION(ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH);
    return onecclSuccess;
}

onecclResult_t CCL_C_API onecclExtractVersionComponents(int versionCode,
                                                        int *major, int *minor,
                                                        int *patch) {
    INIT_CCL;

    *major = versionCode / 10000;
    *minor = (versionCode / 100) % 100;
    *patch = versionCode % 100;
    return onecclSuccess;
}

const char CCL_C_API *onecclGetErrorString(onecclResult_t result) {
    switch (result) {
    // Don't be afraid to add new error types.
    // The more specific error is the better for the user.
    case onecclSuccess:
        return "Success";
    case onecclError:
        return "Generic error";
    case onecclSystemError:
        return "System error";
    case onecclInternalError:
        return "Internal error";
    case onecclInvalidArgument:
        return "Invalid argument";
    case onecclInvalidUsage:
        return "Invalid usage";
    case onecclInProgress:
        return "Operation in progress";
    case onecclFailureGPU:
        return "GPU failure";
    case onecclFailureCPU:
        return "CPU failure";
    case onecclNotImplemented:
        return "Not implemented";
    case onecclAllocFailureCPU:
        return "Could not allocate memory on the CPU";
    case onecclAllocFailureGPU:
        return "Could not allocate memory on the GPU";
    default:
        return "Unknown error code";
    }
}

const char *onecclGetLastError(onecclComm_t /*comm*/) {
    return onecclGetLastErrorString();
}

onecclResult_t onecclGetUniqueId(onecclUniqueId *uniqueId) {
    INIT_CCL;

    auto init_id_func = reinterpret_cast<onecclPluginCallInitId_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_ID));
    if (init_id_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGetUniqueId (INIT_ID)");
        return onecclNotImplemented;
    }

    ONECCL_CHECK_PTR(uniqueId);

    return SAFE_CALL([=]() { return init_id_func(uniqueId); });
}

onecclResult_t onecclCommInitRank(onecclComm_t *comm, size_t nranks,
                                  onecclUniqueId commId, int rank) {
    INIT_CCL;

    onecclConfig_t const config{};
    return onecclCommInitRankConfig(comm, nranks, commId, rank, &config);
}

onecclResult_t onecclCommInitRankConfig(onecclComm_t *comm, size_t nranks,
                                        onecclUniqueId commId, int rank,
                                        const onecclConfig_t *config) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    return init_comm(comm, nranks, commId, rank, config);
}

onecclResult_t onecclCommDestroy(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->destroy == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDestroy");
        return onecclNotImplemented;
    }

    onecclResult_t const result =
        SAFE_CALL([=] { return comm->destroy(comm); });
    if (result != onecclSuccess) {
        return result;
    }

    free(comm);
    return onecclSuccess;
}

onecclResult_t onecclCommSplit(onecclComm_t comm, int color, int key,
                               onecclComm_t *newcomm, onecclConfig_t *config) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->split == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommSplit");
        return onecclNotImplemented;
    }

    // If no config was specified use one from the base comm
    if (config == nullptr) {
        config = &comm->config;
    }

    *newcomm = static_cast<onecclComm_t>(calloc(1, sizeof(struct onecclComm)));
    ONECCL_CHECK_PTR(newcomm);

    (*newcomm)->config = *config;

    if (comm->destroy != nullptr) {
        auto result = SAFE_CALL(
            [=] { return comm->split(comm, color, key, newcomm, config); });
        if (result != onecclSuccess) {
            free(*newcomm);
            *newcomm = nullptr;
            return result;
        }
    }
    return onecclSuccess;
}

onecclResult_t onecclCommUserRank(onecclComm_t comm, int *rank) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(rank);

    if (comm->get_rank == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommUserRank");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->get_rank(comm, rank); });
}

onecclResult_t onecclCommCount(onecclComm_t comm, int *size) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(size);

    if (comm->get_size == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommCount");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->get_size(comm, size); });
}

onecclResult_t onecclAllReduce(void *sendbuff, void *recvbuff, size_t count,
                               onecclDataType_t datatype,
                               onecclRedOp_t reduction_op, onecclComm_t comm,
                               void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->allreduce == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllReduce");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->allreduce(sendbuff, recvbuff, count, datatype,
                               reduction_op, comm, stream);
    });
}

onecclResult_t onecclAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, onecclDataType_t datatype,
                               onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->allgather == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllGather");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->allgather(sendbuff, recvbuff, sendcount, datatype, comm,
                               stream);
    });
}

onecclResult_t CCL_C_API onecclAllToAll(const void *sendbuff, void *recvbuff,
                                        size_t count, onecclDataType_t datatype,
                                        onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->alltoall == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllToAll");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->alltoall(sendbuff, recvbuff, count, datatype, comm,
                              stream);
    });
}

onecclResult_t CCL_C_API onecclAllToAllV(const void *sendbuff,
                                         const size_t *send_counts,
                                         void *recvbuff,
                                         const size_t *recv_counts,
                                         onecclDataType_t datatype,
                                         onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(send_counts);
    ONECCL_CHECK_PTR(recv_counts);

    if (comm->alltoallv == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclAllToAllV");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->alltoallv(sendbuff, send_counts, recvbuff, recv_counts,
                               datatype, comm, stream);
    });
}

onecclResult_t onecclBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, onecclDataType_t datatype,
                               int root, onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->broadcast == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclBroadcast");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->broadcast(sendbuff, recvbuff, count, datatype, root, comm,
                               stream);
    });
}

onecclResult_t onecclReduce(const void *sendbuff, void *recvbuff, size_t count,
                            onecclDataType_t datatype, onecclRedOp_t redop,
                            int root, onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduce == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclReduce");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->reduce(sendbuff, recvbuff, count, datatype, redop, root,
                            comm, stream);
    });
}

onecclResult_t onecclReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, onecclDataType_t datatype,
                                   onecclRedOp_t redop, onecclComm_t comm,
                                   void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduce_scatter == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclReduceScatter");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->reduce_scatter(sendbuff, recvbuff, recvcount, datatype,
                                    redop, comm, stream);
    });
}

onecclResult_t onecclScatter(const void *sendbuff, void *recvbuff, size_t count,
                             onecclDataType_t datatype, int root,
                             onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->scatter == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclScatter");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->scatter(sendbuff, recvbuff, count, datatype, root, comm,
                             stream);
    });
}

onecclResult_t onecclSend(const void *sendbuff, size_t count,
                          onecclDataType_t datatype, int peer,
                          onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->send == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclSend");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->send(sendbuff, count, datatype, peer, comm, stream);
    });
}

onecclResult_t onecclRecv(void *recvbuff, size_t count,
                          onecclDataType_t datatype, int peer,
                          onecclComm_t comm, void *stream) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->recv == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRecv");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->recv(recvbuff, count, datatype, peer, comm, stream);
    });
}

onecclResult_t onecclGroupStart() {
    INIT_CCL;

    auto group_start_func = reinterpret_cast<onecclPluginCallGroupStart_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GROUP_START));
    if (group_start_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGroupStart");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return group_start_func(); });
}

onecclResult_t onecclGroupEnd() {
    INIT_CCL;

    auto group_end_func = reinterpret_cast<onecclPluginCallGroupEnd_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GROUP_END));
    if (group_end_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGroupEnd");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return group_end_func(); });
}

onecclResult_t onecclCommFinalize(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->finalize == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommFinalize");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->finalize(comm); });
}

onecclResult_t onecclCommAbort(onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->abort == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommAbort");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=] { return comm->abort(comm); });
}

onecclResult_t onecclSetDevice(uint32_t index) {
    INIT_CCL;

    auto init_device_func = reinterpret_cast<onecclPluginCallDeviceInit_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_INIT_DEVICE));
    if (init_device_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclSetDevice (INIT_DEVICE)");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return init_device_func(index); });
}

onecclResult_t onecclMemAlloc(void **ptr, size_t size) {
    INIT_CCL;

    ONECCL_CHECK_PTR(ptr);

    auto mem_alloc_func = reinterpret_cast<onecclPluginCallMemAlloc_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_MEM_ALLOC));
    if (mem_alloc_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclMemAlloc");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return mem_alloc_func(ptr, size); });
}

onecclResult_t onecclMemFree(void *ptr) {
    INIT_CCL;

    ONECCL_CHECK_PTR(ptr);

    auto mem_free_func = reinterpret_cast<onecclPluginCallMemFree_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_MEM_FREE));
    if (mem_free_func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclMemFree");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return mem_free_func(ptr); });
}

onecclResult_t onecclCommRegister(onecclComm_t comm, void *buff, size_t size,
                                  void **handle) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(buff);
    ONECCL_CHECK_PTR(handle);

    if (comm->comm_register == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommRegister");
        return onecclNotImplemented;
    }

    return SAFE_CALL(
        [=]() { return comm->comm_register(comm, buff, size, handle); });
}

onecclResult_t onecclCommDeregister(onecclComm_t comm, void *handle) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->comm_deregister == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDeregister");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return comm->comm_deregister(comm, handle); });
}

onecclResult_t onecclCommWindowRegister(onecclComm_t comm, void *buff,
                                        size_t size, onecclWindow_t *window,
                                        onecclWindowFlags_t flags) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(buff);
    ONECCL_CHECK_PTR(window);

    if (comm->comm_window_register == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommWindowRegister");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() {
        return comm->comm_window_register(comm, buff, size, window, flags);
    });
}

onecclResult_t onecclCommWindowDeregister(onecclComm_t comm,
                                          onecclWindow_t window) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    // Allow nullptr window as a no-op, consistent with onecclCommDeregister

    if (comm->comm_window_deregister == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommWindowDeregister");
        return onecclNotImplemented;
    }

    return SAFE_CALL(
        [=]() { return comm->comm_window_deregister(comm, window); });
}

onecclResult_t onecclDevCommCreate(onecclComm_t comm,
                                   const onecclDevCommRequirements_t *reqs,
                                   onecclDevComm_t *dev_comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    ONECCL_CHECK_PTR(reqs);
    ONECCL_CHECK_PTR(dev_comm);

    if (comm->dev_comm_create == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclDevCommCreate");
        return onecclNotImplemented;
    }

    return SAFE_CALL(
        [=]() { return comm->dev_comm_create(comm, reqs, dev_comm); });
}

onecclResult_t onecclDevCommDestroy(onecclComm_t comm,
                                    const onecclDevComm_t *dev_comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);
    // nullptr dev_comm is a no-op, as in onecclCommWindowDeregister.

    if (comm->dev_comm_destroy == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclDevCommDestroy");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return comm->dev_comm_destroy(comm, dev_comm); });
}

onecclResult_t onecclGetPeerDevicePointer(onecclWindow_t window, size_t offset,
                                          int peer, void **out_ptr) {
    INIT_CCL;

    ONECCL_CHECK_PTR(window);
    ONECCL_CHECK_PTR(out_ptr);

    auto func = reinterpret_cast<onecclPluginCallGetPeerDevicePointer_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GET_PEER_DEVICE_POINTER));
    if (func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGetPeerDevicePointer");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return func(window, offset, peer, out_ptr); });
}

onecclResult_t onecclGetLsaDevicePointer(onecclWindow_t window, size_t offset,
                                         int lsa_peer, void **out_ptr) {
    INIT_CCL;

    ONECCL_CHECK_PTR(window);
    ONECCL_CHECK_PTR(out_ptr);

    auto func = reinterpret_cast<onecclPluginCallGetLsaDevicePointer_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GET_LSA_DEVICE_POINTER));
    if (func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGetLsaDevicePointer");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return func(window, offset, lsa_peer, out_ptr); });
}

onecclResult_t onecclGetLocalDevicePointer(onecclWindow_t window, size_t offset,
                                           void **out_ptr) {
    INIT_CCL;

    ONECCL_CHECK_PTR(window);
    ONECCL_CHECK_PTR(out_ptr);

    auto func = reinterpret_cast<onecclPluginCallGetLocalDevicePointer_t>(
        thread_context.plugin.call(ONECCL_PLUGIN_GET_LOCAL_DEVICE_POINTER));
    if (func == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclGetLocalDevicePointer");
        return onecclNotImplemented;
    }

    return SAFE_CALL([=]() { return func(window, offset, out_ptr); });
}

onecclResult_t onecclCommDevice(onecclComm_t comm, int *device) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->create_pre_mul_sum == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclCommDevice");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->get_device(comm, device); });
}

onecclResult_t onecclRedOpCreatePreMulSum(onecclRedOp_t *redop, void *scalar,
                                          onecclDataType_t datatype,
                                          onecclScalarResidence_t residence,
                                          onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->create_pre_mul_sum == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRedOpCreatePreMulSum");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() {
        return comm->create_pre_mul_sum(redop, scalar, datatype, residence,
                                        comm);
    });
}

onecclResult_t onecclRedOpDestroy(onecclRedOp_t redop, onecclComm_t comm) {
    INIT_CCL;

    ONECCL_CHECK_PTR(comm);

    if (comm->reduction_destroy == nullptr) {
        ONECCL_WARN("Plugin does not implement onecclRedOpDestroy");
        return onecclNotImplemented;
    }
    return SAFE_CALL([=]() { return comm->reduction_destroy(redop, comm); });
}
