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

#include "device_buffer.h"

#include "oneapi/ccl.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>

namespace oneccl_test {

namespace {

// Catch OOM at the allocation, in every build type, not at memcpy(nullptr). A
// zero-byte allocation may return nullptr, which is not a failure.
void abort_if_oom(const void *ptr, size_t bytes, const char *what) {
    if (ptr == nullptr && bytes != 0) {
        std::cerr << what << " of " << bytes << " bytes failed\n";
        std::abort();
    }
}

// The owning unique_ptr destructs through a noexcept frame, so the failure must
// not throw. ADD_FAILURE() throws under --gtest_throw_on_failure, so the flag
// is off for the call.
void record_wait_failure(const char *what) {
    const bool throw_on_failure = GTEST_FLAG_GET(throw_on_failure);
    GTEST_FLAG_SET(throw_on_failure, false);
    ADD_FAILURE() << "queue wait failed in ~DeviceBuffer: " << what;
    GTEST_FLAG_SET(throw_on_failure, throw_on_failure);
}

} // namespace

DeviceBuffer::DeviceBuffer(size_t bytes, sycl::queue &queue, Allocator alloc)
    : bytes_(bytes), queue_(&queue), alloc_(alloc) {
    switch (alloc_) {
    case Allocator::ONECCL_SYCL_USM:
        ptr_ = sycl::malloc_device(bytes_, *queue_);
        abort_if_oom(ptr_, bytes_, "sycl::malloc_device");
#ifndef NDEBUG
        // Naming the allocation here catches a residence bug before a
        // collective trips over it. onecclMemAlloc memory may belong to
        // another context, so only the USM path is checked.
        assert(sycl::get_pointer_type(ptr_, queue_->get_context()) ==
               sycl::usm::alloc::device);
#endif
        break;
    case Allocator::ONECCL_MEM_ALLOC: {
        // onecclMemAlloc allocates on the thread's selected device.
        if (bytes_ == 0) {
            break;
        }
        const onecclResult_t code = onecclMemAlloc(&ptr_, bytes_);
        if (code != onecclSuccess) {
            std::cerr << "onecclMemAlloc failed with code "
                      << static_cast<int>(code) << '\n';
            std::abort();
        }
        abort_if_oom(ptr_, bytes_, "onecclMemAlloc");
        break;
    }
    }
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
    // Host-backed: plain host memory, no context to query.
    ptr_ = std::malloc(bytes_);
    abort_if_oom(ptr_, bytes_, "host allocation");
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ == nullptr) {
        return;
    }

    if (!on_device()) {
        std::free(ptr_);
        return;
    }

    // A collective may still read or write this buffer on the queue.
    // A destructor must not throw: a lost device would terminate the process
    // and lose the failure report. Record a failure and leak the buffer.
    try {
        queue_->wait();
    } catch (const std::exception &error) {
        record_wait_failure(error.what());
        return;
    } catch (...) {
        record_wait_failure("unknown exception");
        return;
    }

    switch (alloc_) {
    case Allocator::ONECCL_SYCL_USM:
        sycl::free(ptr_, *queue_);
        break;
    case Allocator::ONECCL_MEM_ALLOC:
        onecclMemFree(ptr_);
        break;
    }
}

void DeviceBuffer::h2d_copy(const HostBuffer &src) {
    if (bytes_ == 0) {
        return;
    }
    if (on_device()) {
        queue_->memcpy(ptr_, src.data(), bytes_).wait();
    } else {
        std::memcpy(ptr_, src.data(), bytes_);
    }
}

HostBuffer DeviceBuffer::d2h_copy() const { return d2h_copy(0, bytes_); }

HostBuffer DeviceBuffer::d2h_copy(size_t byte_offset, size_t bytes) const {
    HostBuffer host(bytes);
    if (on_device()) {
        // The .wait() is the flush: in-order submission orders the collective
        // ahead of this copy, so its completion is the collective's.
        queue_->memcpy(host.data(), ptr_at(byte_offset), bytes).wait();
    } else {
        std::memcpy(host.data(), ptr_at(byte_offset), bytes);
    }
    return host;
}

void *DeviceBuffer::ptr_at(size_t byte_offset) const {
    return static_cast<std::byte *>(ptr_) + byte_offset;
}

} // namespace oneccl_test
