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

/*
 * Regression test for MLSL-4855.
 *
 * graph_integration failed in its ordinary warm-up, before graph recording,
 * on a four-rank PVC int32 allreduce with 16,384 elements. This exact shape
 * selects the fused, multi-workgroup small-allreduce kernel. The kernel copies
 * each rank's input, synchronizes all local work-items, synchronizes the four
 * GPUs, and then reads peer data.
 *
 * Use fresh input values and validate every element after every collective.
 * The CI failure at index 16,128 contained rank 3's previous reduced value
 * plus the current inputs from ranks 0-2. That index belongs to rank 3's
 * reduction slice, and its copy and reduction are performed by different
 * workgroups. This exposes stale temporary-buffer data when a subgroup reports
 * arrival before its stores are flushed. The fix orders every subgroup's cache
 * flush before its contribution to the device-wide barrier.
 */

#include <cstddef>
#include <iostream>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>

#include "oneapi/ccl.hpp"
#include "sycl_base.hpp"

namespace {

constexpr int expected_rank_count = 4;
constexpr std::size_t element_count = 16384;
constexpr std::size_t iteration_count = 10000;
constexpr int iteration_stride = 20000;

// Match graph_integration: one context containing every Level Zero GPU and
// one in-order queue per rank.
std::vector<sycl::device> get_level_zero_gpus() {
    std::vector<sycl::device> devices;
    for (const auto& platform : sycl::platform::get_platforms()) {
        const auto platform_name = platform.get_info<sycl::info::platform::name>();
        if (platform_name.find("Level-Zero") == std::string::npos) {
            continue;
        }
        for (const auto& device : platform.get_devices()) {
            if (device.is_gpu()) {
                devices.push_back(device);
            }
        }
    }
    return devices;
}

} // namespace

int main() {
    ccl::init();
    MPI_Init(nullptr, nullptr);
    atexit(mpi_finalize);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != expected_rank_count) {
        if (rank == 0) {
            std::cerr << "This test requires exactly " << expected_rank_count << " ranks, got "
                      << size << std::endl;
        }
        return 2;
    }

    const auto devices = get_level_zero_gpus();
    if (devices.size() < static_cast<std::size_t>(size)) {
        if (rank == 0) {
            std::cout << "skipped: needs at least " << size << " Level-Zero GPUs, got "
                      << devices.size() << "; # all done" << std::endl;
        }
        return 0;
    }

    const sycl::context sycl_context(devices);
    const sycl::property_list queue_properties{ sycl::property::queue::in_order{} };
    sycl::queue queue(sycl_context, devices[rank % devices.size()], queue_properties);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type kvs_address;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        kvs_address = kvs->get_address();
    }
    MPI_Bcast(kvs_address.data(), kvs_address.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        kvs = ccl::create_kvs(kvs_address);
    }

    auto device = ccl::create_device(queue.get_device());
    auto context = ccl::create_context(queue.get_context());
    auto communicator = ccl::create_communicator(size, rank, device, context, kvs);
    auto stream = ccl::create_stream(queue);

    int* send_buffer = sycl::aligned_alloc_device<int>(4096, element_count, queue);
    int* recv_buffer = sycl::aligned_alloc_device<int>(4096, element_count, queue);
    if (!send_buffer || !recv_buffer) {
        std::cerr << "Rank " << rank << " failed to allocate device buffers" << std::endl;
        return 2;
    }
    std::vector<int> actual(element_count);

    MPI_Barrier(MPI_COMM_WORLD);

    for (std::size_t iteration = 0; iteration < iteration_count; iteration++) {
        const int iteration_offset = static_cast<int>(iteration) * iteration_stride;
        sycl::event fill_event =
            queue.parallel_for(sycl::range<1>(element_count), [=](sycl::id<1> index) {
                send_buffer[index] = static_cast<int>(index[0]) + rank + 1 + iteration_offset;
                recv_buffer[index] = -1;
            });

        std::vector<ccl::event> dependencies;
        dependencies.push_back(ccl::create_event(fill_event));
        auto allreduce_event = ccl::allreduce(send_buffer,
                                              recv_buffer,
                                              element_count,
                                              ccl::datatype::int32,
                                              ccl::reduction::sum,
                                              communicator,
                                              stream,
                                              ccl::default_allreduce_attr,
                                              dependencies);
        allreduce_event.wait();
        queue.memcpy(actual.data(), recv_buffer, element_count * sizeof(int)).wait();

        int local_failure = 0;
        std::size_t failure_index = 0;
        int failure_expected = 0;
        int failure_actual = 0;
        for (std::size_t index = 0; index < element_count; index++) {
            const int expected =
                size * (static_cast<int>(index) + 1 + iteration_offset) + (size * (size - 1)) / 2;
            if (actual[index] != expected) {
                local_failure = 1;
                failure_index = index;
                failure_expected = expected;
                failure_actual = actual[index];
                break;
            }
        }

        if (local_failure) {
            std::cerr << "FAILED: rank=" << rank << ", iteration=" << iteration + 1
                      << ", index=" << failure_index << ", expected=" << failure_expected
                      << ", actual=" << failure_actual << std::endl;
        }

        int global_failure = 0;
        MPI_Allreduce(&local_failure, &global_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_failure) {
            return 1;
        }
    }

    if (!handle_exception(queue)) {
        return 1;
    }

    sycl::free(send_buffer, queue);
    sycl::free(recv_buffer, queue);

    if (rank == 0) {
        std::cout << "PASSED: " << iteration_count << " iterations of 4-rank int32["
                  << element_count << "] fused small allreduce" << std::endl;
    }
    return 0;
}
