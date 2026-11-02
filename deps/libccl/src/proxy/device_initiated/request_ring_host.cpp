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

#include "proxy/device_initiated/request_ring_host.hpp"

#include "common/log/log.hpp"

#include <algorithm>
#include <atomic>
#include <memory>

#include <immintrin.h>

// Write a single 64-byte completion to GDR-mapped device memory, as write_cmd_single() in
// host_initiated/cmd_ring_host.hpp publishes commands. ccl_request_completion is a full cache line
// so MOVDIR64B's 64-byte-aligned destination requirement is met.
__attribute__((target("movdir64b"))) static inline void write_completion_single(
    ccl_request_completion* dst,
    const ccl_request_completion* src) {
    _movdir64b(dst, src);
}

// A channel's tag is the comm's pt2pt tag with the channel index as op_id, and op_id is only 4 bits
// wide (atl_tag.hpp), so 16 channels is the hard ceiling on every tag layout.
static_assert(MAX_P2P_CHANNELS <= 16,
              "a device request's channel is encoded in the ATL tag's op_id field, which is "
              "4 bits wide in mpi_tag_layout and ofi_cxi_tag_layout");

ccl_device_comm_proxy ccl_make_device_comm_proxy(ccl_comm* comm) {
    CCL_THROW_IF_NOT(comm, "device comm proxy: null comm");

    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    CCL_THROW_IF_NOT(atl_comm, "device comm proxy: comm has no atl comm");

    const ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    const ccl_comm_id_t comm_id = static_cast<ccl_comm_id_t>(comm->get_comm_id());

    ccl_device_comm_proxy proxy;
    proxy.atl_comm = atl_comm.get();
    // One tag per channel, the channel index riding in the op_id field - the same trick the pairwise
    // alltoall uses for its chunks (alltoall_pairwise.hpp). rank 0 as in the other pt2pt call sites:
    // ATL matches on the source rank anyway.
    for (uint32_t channel = 0; channel < MAX_P2P_CHANNELS; ++channel) {
        proxy.tags[channel] = atl_comm->tag_creator->create(
            0 /* rank */, comm_id, pt2pt_sched_id, static_cast<ccl_op_id_t>(channel));
    }

    LOG_DEBUG("[proxy] device comm proxy for comm ",
              comm,
              ", atl comm ",
              proxy.atl_comm,
              ", comm_id ",
              comm_id);

    return proxy;
}

void ccl_request_ring_consumer::init(ccl_request_ring* ring,
                                     ccl_request_completions* completions_host) {
    m_ring = ring;
    m_completions_host = completions_host;

    m_scan_cursor = REQUEST_RING_SIZE;
    m_pending.clear();
    m_pending.reserve(REQUEST_RING_SIZE);
    m_live_slots.clear();
    // The structural bound, not max_live_requests: try_post() rejects an already-live slot, so no
    // post ever allocates here however a producer misbehaves.
    m_live_slots.reserve(REQUEST_RING_SIZE);

    resolve_ep_spread();
}

void ccl_request_ring_consumer::complete(const inflight_request& live) {
    // Everything the transfer wrote must be visible before the completion word the kernel polls.
    std::atomic_thread_fence(std::memory_order_release);

    // Staged in a local so the whole line, padding included, is written in one go. MOVDIR64B is a
    // direct store, not held in a write-combining buffer, so it needs no trailing _mm_sfence().
    ccl_request_completion line{};
    line.completion = live.sequence;
    write_completion_single(&m_completions_host->entries[live.slot], &line);
}

/* ATL's own count, so the spread cannot exceed the endpoints that exist; capped at MAX_P2P_CHANNELS
 * because a channel selects one, floored at 1 as a guard on the static attr. Read in init(), which
 * runs from coll_init(): global init has fixed attr.in by then, and the first comm has brought the
 * transport up, so attr.out.mnic_count is the count of NICs actually opened. */
void ccl_request_ring_consumer::resolve_ep_spread() {
    const size_t ep_count = atl_base_comm::attr.in.ep_count;
    const size_t nic_count = atl_base_comm::attr.out.mnic_count;

    m_ep_spread = std::max<size_t>(1, std::min<size_t>(ep_count, MAX_P2P_CHANNELS));

    LOG_INFO("[proxy] device requests spread over ",
             m_ep_spread,
             " endpoint(s) (atl ep_count ",
             ep_count,
             ", nics opened ",
             nic_count,
             ")");

    /* Not an error, but not what a multi-nic configuration asked for: endpoint i maps to nic
     * i % nic_count, so a smaller spread leaves nics idle, a larger one doubles up on them. */
    if (nic_count > 0 && nic_count != m_ep_spread) {
        LOG_WARN("[proxy] endpoint spread ",
                 m_ep_spread,
                 " does not match the ",
                 nic_count,
                 " nic(s) opened, so ",
                 (m_ep_spread < nic_count ? "some nics will stay idle"
                                          : "several endpoints will share a nic"),
                 "; the endpoint count is the larger of the worker endpoints (CCL_WORKER_COUNT, "
                 "times the priority buckets) and the NIC count");
    }
}

uint32_t ccl_request_ring_consumer::sends_inflight(size_t ep) const {
    return static_cast<uint32_t>(
        std::count_if(m_live_slots.begin(), m_live_slots.end(), [this, ep](uint32_t slot_idx) {
            return m_inflight[slot_idx].type == ccl_request_type::send &&
                   m_inflight[slot_idx].ep == ep;
        }));
}

bool ccl_request_ring_consumer::is_live(uint32_t slot_idx) const {
    return std::find(m_live_slots.begin(), m_live_slots.end(), slot_idx) != m_live_slots.end();
}

/* The request the device published as sequence `cursor`, or nullopt if it has not arrived yet -
 * which is also how copy_published() terminates, the reader being strictly in order.
 *
 * The load is volatile because the ring is plain host-pinned memory the device writes behind the
 * compiler's back. The sequence word is published last (see request_ring_write_slot), so seeing
 * `cursor` in it means the payload is in place: what orders the two is the device's system-scope
 * release plus PCIe keeping one requester's posted writes in order, not a formal pairing. The
 * acquire fence covers only the remaining host-side hazard, on x86 a compiler one: it stops the slot
 * copy being hoisted above the flag check. */
std::optional<ccl_device_request> ccl_request_ring_consumer::published_at(uint32_t cursor) const {
    const ccl_device_request& slot = m_ring->slots[slot_of(cursor)];

    if (*static_cast<const volatile uint32_t*>(&slot.sequence) != cursor) {
        return std::nullopt;
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    return slot; // one cache line, copied out while the slot is still ours to read
}

bool ccl_request_ring_consumer::try_post(const ccl_device_request& desc) {
    const uint32_t slot_idx = slot_of(desc.sequence);
    auto* atl_comm = static_cast<atl_base_comm*>(desc.atl_comm);

    // The producer's side of the contract, all checked before anything is posted so a violation
    // cannot leave an untracked request in flight.
    CCL_THROW_IF_NOT(atl_comm,
                     "device request sequence ",
                     desc.sequence,
                     " names no atl comm: the producer was launched with an unset "
                     "ccl_device_comm_proxy");
    CCL_THROW_IF_NOT(desc.channel < MAX_P2P_CHANNELS,
                     "device request sequence ",
                     desc.sequence,
                     " has channel ",
                     desc.channel,
                     " beyond MAX_P2P_CHANNELS ",
                     MAX_P2P_CHANNELS);

    // The endpoint, and through it the nic: the peer derives the same index from the same channel.
    const size_t ep = ep_of(desc.channel);

    // Per endpoint, and sends only - see the header.
    if (desc.req == ccl_request_type::send && sends_inflight(ep) >= max_send_inflight) {
        return false;
    }

    // Can only fire if the producer's back-pressure is broken: it must not reach sequence `seq`
    // before the completion of `seq - REQUEST_RING_SIZE` frees the slot here.
    CCL_THROW_IF_NOT(!is_live(slot_idx),
                     "device request sequence ",
                     desc.sequence,
                     " landed in ring slot ",
                     slot_idx,
                     " while its previous occupant is still in flight");
    // Cannot fire while producers honour the rest of that contract - one request per direction
    // outstanding each, one channel each - which is what bounds the live count.
    CCL_THROW_IF_NOT(m_live_slots.size() < max_live_requests,
                     "device request sequence ",
                     desc.sequence,
                     " would exceed the ",
                     max_live_requests,
                     " requests this consumer can track: a producer has more than one "
                     "request per direction outstanding, or two share a channel");

    // Filled in place and before the post: libfabric is handed a pointer into live.req, so from the
    // post onwards this object must neither move nor be copied.
    inflight_request& live = m_inflight[slot_idx];
    live.req = atl_req_t{};
    live.comm = atl_comm;
    live.slot = slot_idx;
    live.sequence = desc.sequence;
    live.type = desc.req;
    live.ep = ep;

    const atl_status_t status =
        (desc.req == ccl_request_type::send)
            ? atl_comm->send(ep, desc.buf, desc.bytes, desc.peer_rank, desc.tag, live.req)
            : atl_comm->recv(ep, desc.buf, desc.bytes, desc.peer_rank, desc.tag, live.req);

    if (status == ATL_STATUS_AGAIN) {
        // Transport out of resources for now. Nothing is in flight, so `live` is left untracked and
        // the retry of this same request, still pending, overwrites it.
        return false;
    }
    // Not ATL_CALL_THROW_IF_ERROR: that macro declares its own `atl_status_t status`, so handing
    // it a variable of that name makes it self-initialize and report stack garbage.
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS,
                     "device request sequence ",
                     desc.sequence,
                     " (ring slot ",
                     slot_idx,
                     ", channel ",
                     desc.channel,
                     ", endpoint ",
                     ep,
                     ") failed to post: ",
                     atl_status_to_str(status));

    // The one place the live list grows, and only once a transfer really is in flight out of this
    // slot; it shrinks only in publish_completions(), when that transfer lands.
    m_live_slots.push_back(slot_idx);
    return true;
}

/* The type check is here rather than in try_post(): a request of neither type would be skipped by
 * *both* posting passes and the kernel that published it would spin forever instead of seeing a
 * throw. A nonsense channel needs no such check, ep_of() being a modulo: it still lands on one
 * endpoint, and try_post() then throws. */
void ccl_request_ring_consumer::copy_published() {
    while (const std::optional<ccl_device_request> desc = published_at(m_scan_cursor)) {
        CCL_THROW_IF_NOT(desc->req == ccl_request_type::send || desc->req == ccl_request_type::recv,
                         "device request sequence ",
                         m_scan_cursor,
                         " has unexpected type ",
                         static_cast<uint32_t>(desc->req));

        m_pending.push_back(*desc);
        ++m_scan_cursor;
    }
}

void ccl_request_ring_consumer::post_pending(ccl_request_type dir) {
    /* Endpoints that refused a request in *this* pass. Local, so it starts all-false on every call:
     * each sweep asks every endpoint again. Once an endpoint refuses, its later requests are not
     * tried in this pass: an EAGAIN already cost a full retry burst, and a full send cap will not
     * have freed up within the pass. Skipping them also keeps each endpoint posting in publish
     * order. */
    std::array<bool, MAX_P2P_CHANNELS> ep_refused{};

    // True once the request is posted, so remove_if drops it from m_pending.
    const auto post_one = [&](const ccl_device_request& desc) {
        if (desc.req != dir) {
            return false; // the other direction's pass posts it
        }

        const size_t ep = ep_of(desc.channel);
        if (ep_refused[ep]) {
            return false; // waits behind the earlier refused request on this endpoint
        }

        if (!try_post(desc)) {
            ep_refused[ep] = true;
            return false; // stays pending, retried next sweep
        }
        return true;
    };

    // remove_if keeps the remaining entries in order, so each endpoint still posts in publish order.
    // Same remove_if-with-side-effects shape as publish_completions().
    const auto pending_end = std::remove_if(m_pending.begin(), m_pending.end(), post_one);
    m_pending.erase(pending_end, m_pending.end());
}

void ccl_request_ring_consumer::publish_completions() {
    /* check() is the only progress call needed here: under SYCL the ATL progress mode is forced to
     * ATL_PROGRESS_CHECK (atl_ofi_init), where poll() returns immediately and check() drains the
     * completion queue - so probing the posted requests *is* the endpoint progress, and each must be
     * probed on the endpoint it went out on or that endpoint is never progressed. One remove_if pass,
     * retiring each request where it is visited: check() must not see an already completed one. */
    const auto live_end =
        std::remove_if(m_live_slots.begin(), m_live_slots.end(), [this](uint32_t slot_idx) {
            inflight_request& live = m_inflight[slot_idx];

            ATL_CALL_THROW_IF_ERROR(live.comm->check(live.ep, live.req));
            if (!live.req.is_completed) {
                return false;
            }

            // The last thing done with the slot: from this store onwards the device may put a new
            // request in it. Safe before the erase below - nothing else in this sweep reads the
            // list, and the next posting pass cannot start until this one has returned.
            complete(live);
            return true;
        });
    m_live_slots.erase(live_end, m_live_slots.end());
}

bool ccl_request_ring_consumer::progress() {
    if (!m_ring) {
        return false;
    }

    // 1. Read: copy every newly published request out of the ring into m_pending.
    copy_published();

    /* 2. Post: hand the copies to ATL. Every recv first: posting a recv is what lets a peer's capped
     *    sends retire, so a sweep must not leave one waiting behind our own send accounting. Refused
     *    requests stay in m_pending. */
    post_pending(ccl_request_type::recv);
    post_pending(ccl_request_type::send);

    /* 3. Nothing in flight, so nothing to probe. That also means nothing was posted just now -
     *    post_pending() only adds to the live list and only publish_completions() removes. Still
     *    busy, though, while a refused request waits in m_pending for the next sweep. */
    if (m_live_slots.empty()) {
        return !m_pending.empty();
    }

    /* 4. Complete: probe the transfers in flight and publish the completions of those that landed.
     *    Busy even when none did: the completion queue is drained only from inside
     *    publish_completions(), so the caller must not back off while transfers are outstanding. */
    publish_completions();
    return true;
}
