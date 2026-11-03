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

#pragma once

#include "data/host_buffer.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace oneccl_test {

// How the device allocation is obtained.
enum class Allocator : std::uint8_t {
    ONECCL_SYCL_USM,  // sycl::malloc_device on TestEnv's queue
    ONECCL_MEM_ALLOC, // onecclMemAlloc / onecclMemFree
};

// Device memory and both copies (H2D and D2H) on the single queue — the only
// source of a collective's pointers — and a host-backed mode that serves the
// CPU-only plugin.
class DeviceBuffer {
  public:
    DeviceBuffer(size_t bytes, sycl::queue &queue,
                 Allocator alloc = Allocator::ONECCL_SYCL_USM);

    // Host-backed mode, no queue: plain host memory, h2d_copy/d2h_copy become
    // a memcpy. Serves ONECCL_LEGACY_CPU.
    explicit DeviceBuffer(size_t bytes);

    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    // Also deleted: always held via unique_ptr, and a default move would
    // double-free ptr_.
    DeviceBuffer(DeviceBuffer &&) = delete;
    DeviceBuffer &operator=(DeviceBuffer &&) = delete;

    // H2D copy on the single queue (memcpy on the host-backed path).
    void h2d_copy(const HostBuffer &src);

    // D2H copy: whole buffer.
    HostBuffer d2h_copy() const;

    // D2H copy: one window. Copies ptr_at(byte_offset) into an allocation of
    // `bytes`. Bounds peak host memory at two windows for a result far larger
    // than any reason to hold it whole.
    HostBuffer d2h_copy(size_t byte_offset, size_t bytes) const;

    // Device pointer access.
    void *ptr() const { return ptr_; }

    // Device pointer access with an offset.
    void *ptr_at(size_t byte_offset) const;

    // Number of bytes allocated for the device buffer.
    size_t bytes() const { return bytes_; }

    // Whether the buffer is allocated on the device.
    bool on_device() const { return queue_ != nullptr; }

    // SYCL queue associated with the device buffer. Callers must check
    // on_device() first: null on the host-backed path.
    sycl::queue &queue() const {
        assert(queue_ != nullptr && "queue() on a host-backed DeviceBuffer");
        return *queue_;
    }

  private:
    void *ptr_ = nullptr; // Device pointer.
    size_t bytes_ = 0;    // Number of bytes allocated for the device buffer.
    sycl::queue *queue_ =
        nullptr; // SYCL queue associated with the device buffer.
    Allocator alloc_ =
        Allocator::ONECCL_SYCL_USM; // How the device allocation is obtained.
};

} // namespace oneccl_test
