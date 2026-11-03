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
// Remote-memory pointer helpers and the cache-flush inline-asm macro.

#include <array>
#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/consts.hpp"

#define __LscFlushCache() \
    { __asm__ __volatile__("lsc_fence.ugm.evict.gpu" ::: "memory"); }

struct sycl_ptrs_type {
    void *mdfi_ptr_rd{ nullptr }, *mdfi_ptr_wr{ nullptr };
    std::array<void *, MAX_NODE_RANKS> xelink_ptrs_rd{}, xelink_ptrs_wr{};
    std::array<void *, MAX_NODE_RANKS> node_ptrs_rd{}, node_ptrs_wr{};
};

// Peer pointers alias the exchanged buffers, so shifting them by a chunk's byte offset
// lets one IPC exchange serve every chunk. Unset (null) entries stay null.
inline sycl_ptrs_type offset_sycl_ptrs(const sycl_ptrs_type &ptrs, size_t rd_bytes, size_t wr_bytes) {
    auto shift = [](void *p, size_t bytes) {
        return p ? static_cast<void *>(static_cast<char *>(p) + bytes) : p;
    };
    sycl_ptrs_type out;
    out.mdfi_ptr_rd = shift(ptrs.mdfi_ptr_rd, rd_bytes);
    out.mdfi_ptr_wr = shift(ptrs.mdfi_ptr_wr, wr_bytes);
    for (size_t i = 0; i < MAX_NODE_RANKS; i++) {
        out.xelink_ptrs_rd[i] = shift(ptrs.xelink_ptrs_rd[i], rd_bytes);
        out.xelink_ptrs_wr[i] = shift(ptrs.xelink_ptrs_wr[i], wr_bytes);
        out.node_ptrs_rd[i] = shift(ptrs.node_ptrs_rd[i], rd_bytes);
        out.node_ptrs_wr[i] = shift(ptrs.node_ptrs_wr[i], wr_bytes);
    }
    return out;
}
