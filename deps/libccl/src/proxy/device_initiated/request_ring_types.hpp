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

/* Types for the device-initiated request ring (GPU → CPU).
 *
 * The mirror image of proxy/host_initiated/cmd_ring_types.hpp: a SYCL kernel produces send/recv
 * requests into the ring, the CPU proxy thread consumes them by posting ATL transfers and publishes
 * completions back. Memory placement is the inverse too, so in both directions the *consumer* polls
 * memory local to it:
 *
 *   ring (payload + sequence)  -> USM host   (polled by the proxy thread)
 *   completions + next_slot    -> device mem (polled by the kernel, GDR-mmapped so the proxy thread
 *                                 can write it)
 *
 * See request_ring_device.hpp (producer) and request_ring_host.hpp (consumer). */

#pragma once

// Deliberately not proxy/proxy_utils.hpp, unlike the host-initiated counterpart: nothing here needs
// the host wait/backoff helpers, and that header includes host_initiated/cmd_ring_types.hpp, which
// in turn needs symbols from proxy_utils.hpp - so reaching it through this header would break that
// cycle on the wrong side.
#include "proxy/proxy_types.hpp"

#include "coll/algorithms/utils/consts.hpp"
#include "common/utils/utils.hpp"

enum class ccl_request_type : uint32_t {
    nop = 0,
    send = 1,
    recv = 2,
};

/* The communicator as a kernel sees it: what a request needs about its comm and the device cannot
 * compute. Built by ccl_make_device_comm_proxy() (request_ring_host.hpp) and captured by value; the
 * device only copies fields out of it into a slot. Stateless - no registry, nothing to release - so a
 * process may run any number of comms through the ring. The borrowed pointer below stays valid
 * because the caller owns the comm across the launch, and an atl_base_comm is freed only by a comm
 * destroy, which the API requires be preceded by a finalize draining every collective. */
struct ccl_device_comm_proxy {
    // An atl_base_comm*, opaque here so device code needs no ATL types. Never dereferenced on
    // the device, and the ring it lands in is host USM, so it never leaves host memory.
    void* atl_comm = nullptr;
    // One pt2pt ATL tag per channel, a request carrying its own channel's - tag composition is
    // layout-private to ccl_atl_tag_impl. See request_ring_host.hpp for what that buys.
    uint64_t tags[MAX_P2P_CHANNELS] = {};
};

// One request slot, exactly one cache line.
struct ccl_device_request {
    // One direction per request; a rank may publish both, and one flat-ring delivery does exactly
    // that when neither of its ring hops stays in the p2p domain. Also which of the host's two
    // posting passes takes this request (request_ring_host.hpp).
    ccl_request_type req;

    void* buf; // device pointer: source of a send, destination of a recv

    // Copied straight out of the ccl_device_comm_proxy the kernel was launched with - see there
    // for why a host pointer may travel through a ring slot.
    void* atl_comm;
    uint64_t tag;

    uint32_t bytes;
    int32_t peer_rank; // rank in the comm named above

    // Which independent FIFO within the comm this request belongs to, i.e. which per-channel tag `tag`
    // came from, so only the per-channel order has to agree between two ranks. Same channel identity
    // as p2p_barrier_pair_wg()'s, bounded by the same MAX_P2P_CHANNELS. Also what the host derives
    // the ATL endpoint, hence the NIC, from - see request_ring_host.hpp.
    uint32_t channel;

    uint32_t sequence; // published LAST by the device; the host polls only this

    // Pad to a cache line - four words, `buf`'s 8-byte alignment already leaving a hole after `req`.
    uint32_t _pad[4];
};

static_assert(sizeof(ccl_device_request) == CACHELINE_SIZE,
              "ccl_device_request must be exactly one cache line");

constexpr uint32_t REQUEST_RING_SIZE = 64;

// Power-of-two for wrap-safety, not just a cheap modulo: back-pressure computes
// seq - REQUEST_RING_SIZE on a uint32_t meant to wrap, so the seq -> slot mapping survives the 2^32
// wrap only if REQUEST_RING_SIZE divides 2^32. Same reasoning as CMD_RING_SIZE.
static_assert((REQUEST_RING_SIZE & (REQUEST_RING_SIZE - 1)) == 0,
              "REQUEST_RING_SIZE must be a power of two: seq % REQUEST_RING_SIZE must stay "
              "consistent across the uint32_t wrap for the seq - REQUEST_RING_SIZE backpressure");

struct ccl_request_ring {
    ccl_device_request slots[REQUEST_RING_SIZE];

    /* Initialize ring slots to pre-consumed state: slot i gets sequence=i, so the first sequence the
     * device may publish into it is REQUEST_RING_SIZE + i and can never be confused with the seed.
     * Called once at context init - the ring is a monotonic, wrap-safe protocol and must never be
     * reset between ops. */
    static void init_slots(ccl_request_ring* ring) {
        std::memset(ring, 0, sizeof(ccl_request_ring));
        for (uint32_t i = 0; i < REQUEST_RING_SIZE; ++i) {
            ring->slots[i].sequence = i;
        }
    }
};

// Completion tracking, a separate device-memory array. A full cache line per completion: the proxy
// thread publishes it with a single MOVDIR64B, whose destination must be 64-byte aligned, and each
// slot then gets a line of its own for the poll.
struct alignas(CACHELINE_SIZE) ccl_request_completion {
    uint32_t completion; // set by the proxy thread when the transfer is done
    uint32_t _pad[15];
};

static_assert(sizeof(ccl_request_completion) == CACHELINE_SIZE,
              "ccl_request_completion must be exactly one cache line: it is published with "
              "a single MOVDIR64B");

struct ccl_request_completions {
    /* Atomic slot allocator, incremented by the device with fetch_add. It lives here rather than in
     * ccl_request_ring because the ring is in USM host memory, where global_space device atomics are
     * not valid. Monotonic: never reset, grows across all ops. */
    alignas(CACHELINE_SIZE) uint32_t next_slot;

    alignas(CACHELINE_SIZE) ccl_request_completion entries[REQUEST_RING_SIZE];

    /* Initialize to pre-completed state: entry i gets completion=i, matching the slot seeding above,
     * and the allocator starts past the seeds so the first request's backpressure check
     * (completion == seq - REQUEST_RING_SIZE) is already satisfied. */
    static void init_completions(ccl_request_completions* completions) {
        std::memset(completions, 0, sizeof(ccl_request_completions));
        for (uint32_t i = 0; i < REQUEST_RING_SIZE; ++i) {
            completions->entries[i].completion = i;
        }
        completions->next_slot = REQUEST_RING_SIZE;
    }
};

// Kernel-side handle, captured by value: the host-USM ring plus the device-memory completions. Both
// pointers are the *device*-visible ones; the proxy thread has its own aliases.
struct ccl_device_ring_handle {
    ccl_request_ring* ring;
    ccl_request_completions* completions;
};
