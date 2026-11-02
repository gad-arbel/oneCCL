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

/* Host-side consumer for the device-initiated request ring (GPU → CPU).
 *
 * The mirror image of the host-initiated producer (cmd_ring_host.hpp): one read cursor over the
 * ring, copying each request out as the device publishes it, then posting the copies. A sweep does
 * not wait for a transfer: it copies everything published, posts what it can, then probes what is
 * in flight and publishes whichever landed. So the reader stays strictly in order while posting
 * and completion do not.
 *
 * What makes concurrent transfers matchable is the request's `channel`: there is one ATL tag per
 * (comm, channel), so per (peer, tag) FIFO matching pairs a send with the recv of the *same*
 * channel wherever each sits in its rank's ring. Producers on distinct channels may therefore
 * claim slots in any order - what the lock-free allocator in request_ring_device.hpp gives them,
 * and why several work-groups can drive the network at once. The tags are computed by
 * ccl_make_device_comm_proxy(), tag composition being layout-private to ccl_atl_tag_impl; this
 * consumer posts with the tag it is given and never looks a comm up.
 *
 * The producer side of the contract (see request_ring_device.hpp): one request per direction
 * outstanding per producer, one channel per producer, all producers co-resident on the device, and
 * never a claimed sequence left unpublished - the reader cannot skip a hole or wait on an
 * unscheduled producer.
 *
 * THE ENDPOINT IS DERIVED FROM `channel` (ep_of()): the endpoint index is OFI's only NIC selector,
 * and a send matches only a recv on the peer's endpoint of the *same* index - `channel` is the number
 * both ends agree on. The spread is atl_base_comm::attr.in.ep_count, which
 * ccl_executor::generate_atl_attr() floors at the NIC count so every NIC has an endpoint of its own;
 * capped here at MAX_P2P_CHANNELS, a channel selecting one endpoint, and resolved once in init(),
 * which runs from coll_init() - after a comm exists, so ATL is up.
 *
 * SENDS ARE CAPPED PER ENDPOINT, RECVS ARE NOT (max_send_inflight below): measured, not preferred -
 * raising the cap costs bandwidth. Per endpoint so a send queued behind a busy NIC cannot withhold one
 * whose NIC is idle; recvs move nothing until their data arrives.
 * TODO: find out why the transport aggregates worse with more sends in flight; until then, re-measure.
 *
 * READING AND POSTING ARE SEPARATE STEPS. copy_published() copies every published request into
 * m_pending and never calls ATL, so nothing it does can stop it. post_pending() then posts the copies,
 * in publish order per (direction, endpoint); a refusal - the cap, or ATL_STATUS_AGAIN - leaves that
 * copy pending for the next sweep and holds up only its own (direction, endpoint):
 *   - per direction, so the cap cannot deadlock: if a send refused by the cap withheld everything
 *     behind it, recvs included, it would withhold precisely what lets a *peer's* in-flight send
 *     retire, and two ranks each publishing both directions could park each other for good.
 *   - per endpoint, for bandwidth: a send refused by *its* endpoint must not withhold sends bound
 *     for an idle NIC.
 *
 * WHY ONE READER, not a cursor per (direction, endpoint) over the shared ring: such a cursor stops
 * reading whenever it stops posting, and falls behind. The device reuses a slot as soon as its
 * request completes, so the other endpoints' slots ahead of the stopped cursor are completed and
 * rewritten under it: it then finds a sequence it will never match again and wedges for good, or
 * reads a slot mid-rewrite, the sequence word being written last. A reader that only copies reads
 * every slot before its request is even posted, hence before the device may reuse it. Separate
 * *rings* per (direction, endpoint) would do the same, at the price of more allocations and a
 * device-side split.
 *
 * What is left is the transport: a deadlock would need every rank's *recvs* refused on resource
 * exhaustion at once, with at most max_live_requests transfers outstanding - not designed around,
 * only noted.
 *
 * Not negotiable: RDMA only - device pointers go straight to ATL, with no host bounce buffer. */

#pragma once

#include "proxy/device_initiated/request_ring_types.hpp"

#include "atl/atl_base_comm.hpp"
#include "coll/algorithms/utils/consts.hpp"
#include "comm/comm.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

/* Resolve `comm` into the form a device request carries: the atl comm to post on, plus one pt2pt
 * ATL tag per channel. Called once per launch from the algorithm's host thread and stateless by
 * design - no registry, nothing to release, no bound on how many comms may use the ring. The result
 * borrows the comm's atl_base_comm; see ccl_device_comm_proxy for why that needs no keep-alive. */
ccl_device_comm_proxy ccl_make_device_comm_proxy(ccl_comm* comm);

class ccl_request_ring_consumer {
public:
    ccl_request_ring_consumer() = default;

    ccl_request_ring_consumer(const ccl_request_ring_consumer&) = delete;
    ccl_request_ring_consumer& operator=(const ccl_request_ring_consumer&) = delete;

    // `completions` must be the host-visible alias of the device-memory completions array
    // (see ccl_request_ring_context).
    void init(ccl_request_ring* ring, ccl_request_completions* completions_host);

    // One sweep: post every request published since the last one, then probe the transfers in flight
    // and publish the completions of those that landed. Never blocks. Returns true while there is
    // anything to do, including "nothing new, but transfers still in flight" - the caller backs off
    // on false, and in the default ATL_PROGRESS_CHECK mode the completion queue is drained only
    // from inside this call.
    bool progress();

private:
    /* One posted request, keyed by the ring slot it came out of - a collision-free key for free, the
     * producer's back-pressure keeping the device from reusing a slot until the completion of its
     * previous occupant is published. Kept in a fixed array indexed by that slot and written in
     * place, never copied after posting: libfabric keeps a pointer into the atl_req_t it was given,
     * so check() has to see the same object at the same address that send()/recv() got. That is why
     * it is the *index* list below that gets shuffled. The comm, sequence, direction and endpoint are
     * copied in at post time so publish_completions() need not read the ring slot back. */
    struct inflight_request {
        atl_req_t req{};
        atl_base_comm* comm = nullptr;
        uint32_t slot = 0; // ring slot, i.e. the completion entry to publish into
        uint32_t sequence = 0; // what the device polls for
        ccl_request_type type = ccl_request_type::nop;
        // The endpoint it went out on: check() progresses *that* endpoint's completion queue.
        size_t ep = 0;
    };

    // Copy every request published since the last sweep into m_pending, in ring order.
    void copy_published();

    // Post the pending requests of direction `dir`, dropping each one posted from m_pending.
    void post_pending(ccl_request_type dir);

    // The request the device published as sequence `cursor`, if it has arrived yet.
    std::optional<ccl_device_request> published_at(uint32_t cursor) const;

    // Post one pending request. False means "not now" - the send cap or the transport - and leaves
    // it pending for the next sweep.
    bool try_post(const ccl_device_request& desc);

    // Probe every posted request and publish the completion of each one that landed.
    void publish_completions();

    // Publish a completion into device memory through the GDR alias.
    void complete(const inflight_request& live);

    // Both scan the live list rather than being tracked alongside it: it holds at most
    // max_live_requests entries, and each runs at most once per posted request.
    uint32_t sends_inflight(size_t ep) const;
    bool is_live(uint32_t slot_idx) const;

    // Read the endpoint count ATL was initialized with into m_ep_spread, warning if it does not
    // line up with the NICs opened. Called once, from init().
    void resolve_ep_spread();

    // The endpoint, hence the NIC, a request goes out on; both ends derive it from the same channel.
    size_t ep_of(uint32_t channel) const {
        return channel % m_ep_spread;
    }

    static uint32_t slot_of(uint32_t cursor) {
        return cursor % REQUEST_RING_SIZE;
    }

    // Sends in flight *per endpoint*: a send past it stays pending until publish_completions() frees
    // one. Recvs ignore it - see the header. Measured, so re-measure per platform.
    static constexpr uint32_t max_send_inflight = 4;

    /* Requests in flight at once *if producers honour their contract*: one per direction per producer,
     * one producer per channel. A checked expectation, not a cap - reaching it means a producer
     * submitted more than it promised, and throttling would turn that bug into a kernel spinning
     * forever on a completion. The structural bound is REQUEST_RING_SIZE, try_post() rejecting an
     * already-live slot, hence the assert: the expectation must sit inside that bound or the check is
     * dead. */
    static constexpr size_t max_live_requests = 2 * MAX_P2P_CHANNELS;

    ccl_request_ring* m_ring = nullptr;
    ccl_request_completions* m_completions_host = nullptr;

    /* How many endpoints requests are spread over. Set by init() and fixed from then on, so a
     * request keeps its endpoint from pending to in flight. Never 0 once set, so ep_of() is safe on
     * every path progress() can reach. */
    size_t m_ep_spread = 1;

    /* The next sequence the reader will copy, hence the one expected in slot
     * m_scan_cursor % REQUEST_RING_SIZE. Starts at REQUEST_RING_SIZE, slot i being seeded with
     * sequence i as pre-consumed (see ccl_request_ring::init_slots()), monotonic and never reset: a
     * wrap-safe counter, not an index. */
    uint32_t m_scan_cursor = REQUEST_RING_SIZE;

    /* Requests copied out of the ring but not yet posted, in publish order. Copies, so the ring slot
     * is never read again - though the device cannot reuse it before the request completes anyway,
     * which takes a post. Reserved in init() to REQUEST_RING_SIZE, the bound on pending plus live,
     * each slot holding one request at a time - so push_back() never allocates on the proxy
     * thread. */
    std::vector<ccl_device_request> m_pending;

    std::array<inflight_request, REQUEST_RING_SIZE> m_inflight{};

    /* Ring slots of the live requests, and the whole of the in-flight bookkeeping: a slot enters once
     * a transfer really is in flight out of it and leaves when that transfer lands, so this list
     * alone answers "what is in flight". Unordered, completions arriving in any order, and short.
     * Reserved in init() to REQUEST_RING_SIZE - an unconditional bound given try_post()'s
     * slot-uniqueness check - so push_back() never allocates on the proxy thread. */
    std::vector<uint32_t> m_live_slots;
};
