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

#include "oneapi/ccl.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

constexpr int root = 0;
constexpr std::size_t KiB = 1024;
constexpr std::size_t MiB = 1024 * KiB;
constexpr std::size_t auto_fallback_message_bytes = 4 * MiB;
constexpr std::size_t large_message_bytes = 32 * MiB;
constexpr std::size_t chunking_slice_bytes = 4 * MiB;
constexpr std::size_t chunking_full_slice_count = 4;
constexpr std::size_t chunking_message_bytes =
    chunking_full_slice_count * chunking_slice_bytes + sizeof(int);

struct broadcast_case {
    std::size_t count;
    bool in_place;
    const char *name;
};

constexpr std::size_t element_count(std::size_t bytes) {
    return bytes / sizeof(int);
}

int expected_value(std::size_t index) {
    return static_cast<int>((index * 17 + 11) % 251);
}

sycl::queue create_queue(int local_rank) {
    for (const auto &platform : sycl::platform::get_platforms()) {
        if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
            continue;
        }

        const auto devices = platform.get_devices();
        if (local_rank >= 0 &&
            static_cast<std::size_t>(local_rank) < devices.size()) {
            return sycl::queue(devices[local_rank],
                               sycl::property::queue::in_order{});
        }
    }

    throw std::runtime_error(
        "Level Zero GPU for the local MPI rank was not found");
}

void require_env(const char *name, const char *expected) {
    const char *actual = std::getenv(name);
    if (expected == nullptr) {
        if (actual != nullptr) {
            throw std::runtime_error(std::string(name) + " must be unset");
        }
        return;
    }

    if (actual == nullptr || expected != std::string(actual)) {
        throw std::runtime_error(std::string(name) + " must be set to " +
                                 expected);
    }
}

void require_env_size(const char *name, std::size_t expected) {
    const std::string expected_string = std::to_string(expected);
    require_env(name, expected_string.c_str());
}

std::vector<broadcast_case> get_cases(const std::string &scenario) {
    if (scenario == "auto_fallback") {
        require_env("CCL_SYCL_BROADCAST_RING", "1");
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "1");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE", "1");
        return {
            {element_count(auto_fallback_message_bytes), false,
             "auto-fallback/out-of-place"},
        };
    }
    if (scenario == "auto_ring") {
        require_env("CCL_SYCL_BROADCAST_RING", "1");
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "1");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE", "1");
        return {
            {element_count(large_message_bytes), false, "large/out-of-place"},
            {element_count(large_message_bytes), true, "large/in-place"},
        };
    }
    if (scenario == "bmg_default") {
        require_env("CCL_SYCL_BROADCAST_RING", nullptr);
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "0");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE", "1");
        return {
            {element_count(large_message_bytes), false, "large/out-of-place"},
            {element_count(large_message_bytes), true, "large/in-place"},
        };
    }
    if (scenario == "chunking") {
        require_env("CCL_SYCL_BROADCAST_RING", "1");
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "0");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE", "1");
        require_env_size("CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD",
                         chunking_slice_bytes);
        return {
            {element_count(chunking_message_bytes), false,
             "threshold-tail/out-of-place"},
        };
    }
    if (scenario == "copy_kernel") {
        require_env("CCL_SYCL_BROADCAST_RING", "1");
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "0");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE", "0");
        require_env("CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD", "0");
        return {
            {element_count(large_message_bytes), false, "large/out-of-place"},
            {element_count(large_message_bytes), true, "large/in-place"},
        };
    }
    if (scenario == "relay_copy_engine" || scenario == "relay_copy_kernel") {
        const bool use_copy_engine = scenario == "relay_copy_engine";
        require_env("CCL_SYCL_BROADCAST_RING", "1");
        require_env("CCL_SYCL_BROADCAST_RING_AUTO", "0");
        require_env("CCL_SYCL_BROADCAST_RING_COPY_ENGINE",
                    use_copy_engine ? "1" : "0");
        require_env("CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD", "0");
        return {
            {element_count(large_message_bytes), false, "relay/out-of-place"},
            {element_count(large_message_bytes), true, "relay/in-place"},
        };
    }

    throw std::runtime_error("unknown scenario: " + scenario);
}

bool run_broadcast_case(sycl::queue &queue, onecclComm_t comm, int rank,
                        const broadcast_case &test_case) {
    std::vector<int> input(test_case.count, -1);
    if (rank == root) {
        for (std::size_t index = 0; index < test_case.count; index++) {
            input[index] = expected_value(index);
        }
    }

    int *send_buffer = sycl::malloc_device<int>(test_case.count, queue);
    int *recv_buffer = test_case.in_place
                           ? send_buffer
                           : sycl::malloc_device<int>(test_case.count, queue);
    if (send_buffer == nullptr || recv_buffer == nullptr) {
        if (recv_buffer != nullptr && recv_buffer != send_buffer) {
            sycl::free(recv_buffer, queue);
        }
        if (send_buffer != nullptr) {
            sycl::free(send_buffer, queue);
        }
        throw std::runtime_error("device buffer allocation failed");
    }

    queue.memcpy(send_buffer, input.data(), test_case.count * sizeof(int))
        .wait();
    if (!test_case.in_place) {
        queue.fill(recv_buffer, -1, test_case.count).wait();
    }

    const onecclResult_t result =
        onecclBroadcast(send_buffer, recv_buffer, test_case.count, onecclInt32,
                        root, comm, &queue);

    bool passed = result == onecclSuccess;
    std::vector<int> output(test_case.count);
    if (passed) {
        queue.memcpy(output.data(), recv_buffer, test_case.count * sizeof(int))
            .wait();
        for (std::size_t index = 0; index < test_case.count; index++) {
            if (output[index] != expected_value(index)) {
                std::cerr << "Rank " << rank << ": " << test_case.name
                          << " mismatch at index " << index << ", expected "
                          << expected_value(index) << ", got " << output[index]
                          << '\n';
                passed = false;
                break;
            }
        }
    } else {
        std::cerr << "Rank " << rank << ": onecclBroadcast returned " << result
                  << " for " << test_case.name << '\n';
    }

    if (recv_buffer != send_buffer) {
        sycl::free(recv_buffer, queue);
    }
    sycl::free(send_buffer, queue);
    return passed;
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 2 || world_size < 2) {
        if (rank == root) {
            std::cerr << "Usage: broadcast_ring_sycl <scenario>, launched with "
                         "at least two MPI ranks\n";
        }
        MPI_Finalize();
        return 1;
    }

    try {
        const std::string scenario = argv[1];
        if ((scenario == "relay_copy_engine" ||
             scenario == "relay_copy_kernel") &&
            world_size < 3) {
            throw std::runtime_error(
                "relay scenarios require at least three MPI ranks");
        }
        const auto test_cases = get_cases(scenario);

        MPI_Comm local_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &local_comm);
        int local_rank = 0;
        MPI_Comm_rank(local_comm, &local_rank);

        onecclUniqueId unique_id{};
        int unique_id_ok = 1;
        if (rank == root && onecclGetUniqueId(&unique_id) != onecclSuccess) {
            unique_id_ok = 0;
        }
        MPI_Bcast(&unique_id_ok, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (!unique_id_ok) {
            throw std::runtime_error("onecclGetUniqueId failed");
        }
        MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, root,
                  MPI_COMM_WORLD);

        if (onecclSetDevice(local_rank) != onecclSuccess) {
            throw std::runtime_error("onecclSetDevice failed");
        }

        onecclComm_t comm = nullptr;
        if (onecclCommInitRank(&comm, world_size, unique_id, rank) !=
            onecclSuccess) {
            throw std::runtime_error("onecclCommInitRank failed");
        }

        sycl::queue queue = create_queue(local_rank);
        bool all_passed = true;
        for (const auto &test_case : test_cases) {
            const bool local_passed =
                run_broadcast_case(queue, comm, rank, test_case);
            int local_status = local_passed ? 1 : 0;
            int global_status = 0;
            MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            if (!global_status) {
                all_passed = false;
                break;
            }
            if (rank == root) {
                std::cout << "PASS: " << scenario << " " << test_case.name
                          << '\n';
            }
        }

        onecclCommDestroy(comm);
        MPI_Comm_free(&local_comm);
        MPI_Finalize();
        return all_passed ? 0 : 1;
    } catch (const std::exception &exception) {
        std::cerr << "Rank " << rank << ": " << exception.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
