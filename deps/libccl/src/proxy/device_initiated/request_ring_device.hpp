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

/* Device-side producer for the request ring (GPU → CPU).
 *
 * A kernel that needs an ATL send/recv claims a ring slot, writes the request into USM host memory,
 * and spins on the matching completion word in device memory until the proxy thread publishes it.
 * The kernel is never torn down for the transfer.
 *
 * Ordering rule that makes the host's single-word poll safe: payload fields first, then a release
 * fence, then the sequence word. The host-initiated side gets this atomically from the 64-byte
 * _movdir64b store; the device has no 64-byte atomic store, so the fence replaces it - the same
 * idiom cmd_ring_complete() already uses to publish to host USM. */

#pragma once

#include "proxy/device_initiated/request_ring_types.hpp"

#include <sycl/sycl.hpp>

// Claim the next sequence. `next_slot` lives in device memory, so a global_space atomic is valid here
// (unlike anything in the host-USM ring), and device scope is enough - only the device increments it.
static inline uint32_t request_ring_next_seq(ccl_request_completions *completions) {
    sycl::atomic_ref<uint32_t,
                     sycl::memory_order::acq_rel,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        slot_atomic(completions->next_slot);
    return slot_atomic.fetch_add(1);
}

// Wrap-safe monotonic wait: spins until the completion word has advanced to at least `target`, the
// same "before" comparison as the host's wait_backoff_ge(). The word lives in device memory but the
// host writes it through the GDR alias, so the load needs a system-scope atomic to be coherent.
static inline void request_ring_wait_completion_ge(ccl_request_completions *completions,
                                                   uint32_t slot_idx,
                                                   uint32_t target) {
    sycl::atomic_ref<uint32_t,
                     sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        completion_atomic(completions->entries[slot_idx].completion);

    while (static_cast<int32_t>(completion_atomic.load() - target) < 0) {
    }
}

static inline void request_ring_write_slot(ccl_device_ring_handle handle,
                                           uint32_t seq,
                                           ccl_request_type req,
                                           void *buf,
                                           uint32_t bytes,
                                           int32_t peer_rank,
                                           const ccl_device_comm_proxy &comm,
                                           uint32_t channel) {
    const uint32_t slot_idx = seq % REQUEST_RING_SIZE;
    ccl_device_request *slot = &handle.ring->slots[slot_idx];

    // Back-pressure: the slot is reusable only once the request that occupied it one lap ago has
    // completed. The seeding (slot i -> sequence i, completion i) makes the first lap trivial.
    request_ring_wait_completion_ge(handle.completions, slot_idx, seq - REQUEST_RING_SIZE);

    slot->req = req;
    slot->buf = buf;
    slot->bytes = bytes;
    slot->peer_rank = peer_rank;
    slot->atl_comm = comm.atl_comm;
    slot->tag = comm.tags[channel];
    slot->channel = channel;

    // Publish: everything above must be visible to the host before the sequence word it polls on,
    // and the sequence store must then leave the device.
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    slot->sequence = seq;
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

static inline uint32_t request_ring_submit_send(ccl_device_ring_handle handle,
                                                void *src,
                                                uint32_t bytes,
                                                int32_t dest_rank,
                                                const ccl_device_comm_proxy &comm,
                                                uint32_t channel) {
    const uint32_t seq = request_ring_next_seq(handle.completions);
    request_ring_write_slot(
        handle, seq, ccl_request_type::send, src, bytes, dest_rank, comm, channel);
    return seq;
}

static inline uint32_t request_ring_submit_recv(ccl_device_ring_handle handle,
                                                void *dst,
                                                uint32_t bytes,
                                                int32_t src_rank,
                                                const ccl_device_comm_proxy &comm,
                                                uint32_t channel) {
    const uint32_t seq = request_ring_next_seq(handle.completions);
    request_ring_write_slot(
        handle, seq, ccl_request_type::recv, dst, bytes, src_rank, comm, channel);
    return seq;
}

// Block until the proxy thread has completed the request published as `seq`. The acquire fence
// afterwards is what makes the NIC-written payload visible to this kernel's subsequent loads.
static inline void request_ring_wait(ccl_device_ring_handle handle, uint32_t seq) {
    request_ring_wait_completion_ge(handle.completions, seq % REQUEST_RING_SIZE, seq);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
}

/* Group-granularity wrappers, the intended way for a kernel to use the ring: the group's leader
 * submits and spins, and the group is released together, so nothing synchronises wider than the
 * group. Templated on the group type so one definition serves work-group and sub-group callers. The
 * sequence is returned on the leader only - request_ring_wait_group() is its only consumer.
 *
 * Two obligations on the caller:
 *   - sycl::group_barrier(g) before submitting a send whose payload the whole group wrote, so those
 *     stores precede the release fence inside request_ring_write_slot().
 *   - one request per direction outstanding per producer, and a channel of its own per producer: each
 *     request carries the tag of its (comm, channel), so two producers sharing a channel would be
 *     indistinguishable to tag matching - while producers on distinct channels may claim slots in
 *     any order, which is the point of the lock-free allocator above. A send and a recv may be
 *     outstanding together, the host draining the two directions independently. */

template <typename Group>
static inline uint32_t request_ring_submit_send_group(Group g,
                                                      ccl_device_ring_handle handle,
                                                      void *src,
                                                      uint32_t bytes,
                                                      int32_t dest_rank,
                                                      const ccl_device_comm_proxy &comm,
                                                      uint32_t channel) {
    if (g.leader()) {
        return request_ring_submit_send(handle, src, bytes, dest_rank, comm, channel);
    }
    return 0;
}

template <typename Group>
static inline uint32_t request_ring_submit_recv_group(Group g,
                                                      ccl_device_ring_handle handle,
                                                      void *dst,
                                                      uint32_t bytes,
                                                      int32_t src_rank,
                                                      const ccl_device_comm_proxy &comm,
                                                      uint32_t channel) {
    if (g.leader()) {
        return request_ring_submit_recv(handle, dst, bytes, src_rank, comm, channel);
    }
    return 0;
}

// `seq` only has to be valid on the leader - exactly what the submit wrappers return.
template <typename Group>
static inline void request_ring_wait_group(Group g, ccl_device_ring_handle handle, uint32_t seq) {
    if (g.leader()) {
        request_ring_wait(handle, seq);
    }
    // Releases the group once the transfer is done. The leader's acquire fence inside
    // request_ring_wait() orders only its own loads, so the rest of the group needs one of its own -
    // the same shape as the trailing lfence_system() in p2p_barrier_pair().
    sycl::group_barrier(g);
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
}
