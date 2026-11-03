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

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

// SYCL gather reg-test. Uses an in-order queue + USM buffers so it actually runs the device
// kernels (gather_small / gather_large) instead of the scheduler path the functional suite hits.
//
// Gather: every rank contributes count elements; root receives size*count with rank i's block at
// recv_buf + i*count. Non-root ranks don't use recv_buf.
//
// The .sh driver sweeps message size, CCL_SYCL_GATHER_LARGE_TMP_BUF and
// CCL_SYCL_GATHER_LARGE_PROTOCOL to cover all kernel variants, and what non-root
// passes as recv_buf: nullptr, or plain heap memory, which the API equally allows.

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage ./gather_usm_test [device] [alloc_type] [nonroot_recv]\n";
        cout << "device could be 'cpu' or 'gpu'\n";
        cout << "nonroot_recv could be 'null' (default) or 'host'\n";
        cout << "example: ./gather_usm_test gpu device host\n";
        exit(1);
    }

    string device_name = argv[1];

    if (device_name != "cpu" && device_name != "gpu") {
        cout << "error: Invalid device '" << device_name << "'.\n";
        cout << "device must be either 'cpu' or 'gpu'.\n";
        exit(1);
    }

    size_t count = 10 * 1024 * 1024;

    const int root = 0;
    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    sycl::queue q;
    if (!create_test_sycl_queue(device_name, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::shared;
    if (argc > 2) {
        usm_alloc_type = usm_alloc_type_from_string(argv[2]);
    }

    const string nonroot_recv = (argc > 3 && argv[3][0] != '-') ? argv[3] : "null";
    if (nonroot_recv != "null" && nonroot_recv != "host") {
        cout << "error: Invalid nonroot_recv '" << nonroot_recv << "'.\n";
        exit(1);
    }

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

    /* Every rank contributes count; only root needs the full receive buffer. */
    auto send_buf = allocator.allocate(count, usm_alloc_type);
    vector<int> nonroot_host_recv(1);
    int *recv_buf = (rank == root)             ? allocator.allocate(count * size, usm_alloc_type)
                    : (nonroot_recv == "host") ? nonroot_host_recv.data()
                                               : nullptr;

    const size_t check_count = (rank == root) ? count * size : 1;
    sycl::buffer<int> check_buf(check_count);

    /* Rank i tags element id with (i*7 + id%256); root expects that tag in block i. */
    auto e = q.submit([&](auto &h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = static_cast<int>((rank * 7 + (id % 256)) % 100000);
        });
    });

    /* poison the receive buffer so a missing write is caught too */
    if (rank == root) {
        e = q.submit([&](auto &h) {
            h.depends_on(e);
            h.parallel_for(count * size, [=](auto id) { recv_buf[id] = -1; });
        });
    }

    /* do not wait completion of kernel and provide it as dependency for operation */
    vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));

    /* invoke gather */
    auto attr = ccl::create_operation_attr<ccl::gather_attr>();
    ccl::gather(send_buf, recv_buf, count, ccl::datatype::int32, root, comm, stream, attr, deps)
        .wait();

    /* open recv_buf and check its correctness on the device side */
    if (rank == root) {
        q.submit([&](auto &h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            h.parallel_for(count * size, [=](auto id) {
                const size_t block = id / count;
                const size_t elem = id % count;
                const int expected = static_cast<int>((block * 7 + (elem % 256)) % 100000);
                check_buf_acc[id] = (recv_buf[id] == expected) ? 0 : -1;
            });
        });
    }

    if (!handle_exception(q))
        return -1;

    if (rank == root) {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        size_t i;
        for (i = 0; i < count * size; i++) {
            if (check_buf_acc[i] == -1) {
                cout << "FAILED at index " << i << "\n";
                return -1;
            }
        }
    }

    cout << "PASSED\n";

    return 0;
}
