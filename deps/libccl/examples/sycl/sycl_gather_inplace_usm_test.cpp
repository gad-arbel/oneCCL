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

// In-place USM gather: at root send_buf aliases recv_buf + root*count*dsize, so
// is_gather_inplace() is true and gather_large_impl_ipc skips root's local
// memcpy. The small kernel has no in-place branch by construction — it stages
// root's block like any other and writes it back where it came from — so this
// test is what guarantees that absence of a special case is actually correct.
//
// Peers pass recv_buf == nullptr, covering that contract too.
//
// Not covered here: out-of-place (sycl_gather_usm_test.cpp), the sycl::buffer
// overload (sycl_gather_test.cpp), sub-communicators
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

    /* root aliases send_buf into its own slot of recv_buf; peers pass nullptr
     * recv_buf, since only root supplies one */
    int *send_buf = nullptr;
    int *recv_buf = nullptr;
    if (rank == (int)root_rank) {
        recv_buf = allocator.allocate(count * size, usm_alloc_type);
        send_buf = recv_buf + root_rank * count;
    }
    else {
        send_buf = allocator.allocate(count, usm_alloc_type);
    }

    /* do not wait completion of kernel and provide it as dependency for operation */
    vector<ccl::event> deps;

    if (rank == (int)root_rank) {
        /* poison the whole recv buffer, then plant root's contribution in its
         * slot — the two overlap, which is the point of the test */
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count * size, [=](auto id) { recv_buf[id] = -1; });
        });
        auto e2 = q.submit([&](auto &h) {
            h.depends_on(e);
            h.parallel_for(count, [=](auto id) { send_buf[id] = 1; });
        });
        deps.push_back(ccl::create_event(e2));
    }
    else {
        const int fill = rank + 1;
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count, [=](auto id) { send_buf[id] = fill; });
        });
        deps.push_back(ccl::create_event(e));
    }

    /* invoke gather (in-place on root) */
    auto attr = ccl::create_operation_attr<ccl::gather_attr>();
    ccl::gather(send_buf, recv_buf, count, root_rank, comm, stream, attr, deps).wait();

    /* check root holds block (i+1) at offset i*count */
    if (rank != (int)root_rank) {
        std::cout << "PASSED\n";
        return 0;
    }

    sycl::buffer<int> check_buf(count * size);
    q.submit([&](auto &h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count * size, [=](auto id) {
            const int expected = static_cast<int>(id / count) + 1;
            check_buf_acc[id] = (recv_buf[id] != expected) ? -1 : 0;
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
