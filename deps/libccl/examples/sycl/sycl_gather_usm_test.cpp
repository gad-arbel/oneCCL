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

// Out-of-place USM gather: separate send_buf and recv_buf, so
// is_gather_inplace() is false and root does a real copy of its own block.
//
// Not covered here: in-place (sycl_gather_inplace_usm_test.cpp), the
// sycl::buffer overload (sycl_gather_test.cpp), sub-communicators
// (sycl_gather_usm_comm_split.cpp).

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    if (!check_example_args(argc, argv))
        exit(1);

    const size_t root_rank = 0;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);
    const size_t count = args.count;

    string device_type = argv[1];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm_alloc_type_from_string(argv[2]);

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* every rank contributes count, root receives size*count */
    auto send_buf = allocator.allocate(count, usm_alloc_type);
    auto recv_buf = allocator.allocate(count * size, usm_alloc_type);

    /* do not wait completion of kernel and provide it as dependency for operation */
    vector<ccl::event> deps;

    {
        /* rank r sends (r + 1), so root's block i must read (i + 1) */
        const int fill = rank + 1;
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count, [=](auto id) { send_buf[id] = fill; });
        });
        deps.push_back(ccl::create_event(e));
    }

    {
        /* poison recv_buf to catch under-writes */
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count * size, [=](auto id) { recv_buf[id] = -1; });
        });
        deps.push_back(ccl::create_event(e));
    }

    auto attr = ccl::create_operation_attr<ccl::gather_attr>();
    ccl::gather(send_buf, recv_buf, count, root_rank, comm, stream, attr, deps).wait();

    if (rank != (int)root_rank) {
        std::cout << "PASSED\n";
        return 0;
    }

    sycl::buffer<int> check_buf(count * size);
    q.submit([&](auto &h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count * size, [=](auto id) {
            const int expected = static_cast<int>(id / count) + 1;
            if (recv_buf[id] != expected) {
                check_buf_acc[id] = -1;
            }
            else {
                check_buf_acc[id] = 0;
            }
        });
    });

    if (!handle_exception(q))
        return -1;

    /* print out the result of the test on the host side */
    sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
    size_t i;
    for (i = 0; i < count * size; i++) {
        if (check_buf_acc[i] == -1) {
            std::cout << "FAILED\n";
            break;
        }
    }
    if (i == count * size) {
        std::cout << "PASSED\n";
    }

    return 0;
}
