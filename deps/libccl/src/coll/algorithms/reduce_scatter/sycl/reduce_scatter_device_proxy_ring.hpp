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

/* Flat reduce-scatter: one SYCL kernel runs the whole N-rank ring over the user communicator,
 * with no scale-up / scale-out decomposition as in reduce_scatter_sycl_multi_domain(). Each hop
 * picks its transport from where the neighbour sits: IPC write into its tmp buffer inside the
 * local p2p domain, in-kernel ATL send/recv (proxy/device_initiated/) beyond it.
 *
 * Selected by CCL_SYCL_REDUCE_SCATTER_DOMAIN_MODE=flat, and RDMA-only. The rank layout is free:
 * ring position is the global-comm rank, and the kernel derives from it which block it forwards, so
 * no rearrange step is needed whatever order the domains sit in. Each rank resolves its two hops
 * independently - both may be local, one remote, or both, the last being what one rank per p2p domain
 * gives - and the hops agree pairwise because domain membership is symmetric. Nothing synchronises
 * wider than a work-group: the entry rendezvous is a separate launch (see the driver) and
 * per-delivery handshakes are p2p_barrier_pair_wg(). */

#pragma once

// reduce_at / reduce_copy_at / copy_at: index explicitly rather than by global id, as a
// per-channel range needs.
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

#include "proxy/proxy.hpp"
#include "proxy/device_initiated/request_ring_device.hpp"

/* Vocabulary:
 *   block          one rank's share of the message, recv_count elements
 *   channel        one work-group; owns a fixed element range of every block and a fixed byte
 *                  range inside every staging slot, both for the whole call
 *   chunk          one trip of the chunk_idx loop, sliding along that element range
 *   delivery       one transfer to the right neighbour; a chunk runs N-1, the index counting on
 *                  across chunks rather than restarting
 *   staging slot   the region of sycl_tmp_buf holding one delivery, wrapping modulo the slot
 *                  count below (the proxy request ring has its own "slots")
 *
 * A ch_ prefix marks a per-channel value, so the ragged tail is clamped per channel; note
 * staging slot size == num_channels * chunk_size, up to align_bytes rounding. */

/* Channels the ring is spread over, one work-group each. Not an env var: it decides how many
 * requests a rank publishes, so every rank must agree on it. Must stay <= MAX_P2P_CHANNELS,
 * which sizes the pairwise-barrier cells and keeps every channel co-resident, as the proxy
 * ring's in-order reader needs. Pairs with max_send_inflight in request_ring_host.hpp,
 * the half that saturates first; empirically tuned, so re-measure both on a new platform. */
constexpr size_t reduce_scatter_device_proxy_num_channels = 8;

static_assert(reduce_scatter_device_proxy_num_channels <= MAX_P2P_CHANNELS,
              "more channels than the pairwise-barrier cells coll_init() allocates");

// Work floor per channel; below it the extra requests and barriers cost more than the bandwidth
// they add, so the channel count drops (at one channel this is the single work-group path).
constexpr size_t reduce_scatter_device_proxy_min_bytes_per_channel = 16 * 1024;

/* Staging slots the tmp region is cut into, i.e. the delivery pipeline depth; also sets the slot
 * size and hence how many chunks a channel runs. A fixed count rather than one derived from the
 * ring size: that keeps the chunk count a function of message size alone, and avoids a node-local
 * count that ranks on differently populated nodes would not agree on.
 *
 * Four is the floor: only our left neighbour writes our slots, and it cannot run more than one
 * delivery ahead of us either way that hop is carried. Locally, p2p_barrier_pair_wg() bounds the pair
 * to one delivery of drift (see _sync()); remotely, tighter still, the NIC writes our slot for
 * delivery d only because we posted that recv within delivery d and waited for it before the delivery
 * ended, so the peer cannot run ahead into a slot we have not asked for. With a rank at delivery d
 * reading slot d-1 and writing slot d, three slots then keep a write off the slot a peer is reading
 * within a chunk and a fourth covers the boundary, where the peeled last arrival runs no barrier.
 * Both bounds are pairwise, hence independent of ring and domain size, and both need the delivery
 * index continuous across chunks - see delivery_base in the kernel. */
constexpr int reduce_scatter_device_proxy_staging_slots = 4;

// Alignment every chunk boundary and per-channel byte range is cut on. Same knob the large tmp
// buffers themselves are cut on (sycl_coll_base.cpp), and a multiple of the vector width.
inline size_t reduce_scatter_device_proxy_align_bytes() {
    return ccl::global_data::env().kernel_mem_align;
}

/* Bytes of one staging slot. The large tmp buffers are one contiguous sycl_tmp_buf_size allocation
 * (sycl_coll_base.cpp), cut here into equal slots and ignoring the ccl_large_tmp_bufs::buf_count
 * split. Rounding down keeps every slot aligned (the base is CCL_REG_MSG_ALIGNMENT-aligned); slots
 * are contiguous, so this is their stride too. Zero means the tmp buffer is too small for this
 * path - can_use() rejects that. */
inline size_t reduce_scatter_device_proxy_staging_slot_size() {
    const size_t align_bytes = reduce_scatter_device_proxy_align_bytes();
    return ccl::global_data::env().sycl_tmp_buf_size / reduce_scatter_device_proxy_staging_slots / align_bytes *
           align_bytes;
}

// The ring, resolved on the host.
struct reduce_scatter_device_proxy_topology {
    int ring_size = 0; // N: global_comm->size()
    int ring_rank = 0; // this rank in global_comm
    int left_rank = 0; // global_comm rank we receive from
    int right_rank = 0; // global_comm rank we send to
    // Ranks inside p2p_comm, or -1 when that hop leaves the local p2p domain.
    int local_left_peer = -1;
    int local_right_peer = -1;
    // Filled here so the per-dtype driver template never touches the proxy service.
    ccl_device_ring_handle ring{};
    // The comm every request of this launch names, and its per-channel tags.
    ccl_device_comm_proxy comm{};
};

/* Work division. Every field must be identical on every rank - they decide how many requests
 * each channel publishes, and a channel's k-th send is matched with the same channel's k-th recv
 * on the next rank, so a disagreement hangs. Hence only message size, dtype size and the
 * (env-derived, comm-uniform) staging slot size feed this, never a device property. Values the
 * host alone needs - work-group sizing, logging - are derived in the driver instead. */
struct reduce_scatter_device_proxy_geometry {
    size_t num_channels = 1;
    size_t align_elems = 1; // elements per alignment unit; every boundary here is one
    size_t num_align_units = 0; // alignment units in this rank's output block
    size_t chunk_count = 0; // elements one channel moves per chunk, at most

    struct ch_range {
        size_t beg = 0;
        size_t end = 0;

        size_t count() const {
            return end - beg;
        }
    };

    /* The element range of a block that channel `channel` owns for the whole call, cut on
     * align_elems boundaries so can_use_full_vector()'s guarantee survives the split. The one
     * definition of that split: the kernel indexes with it, the host sizes work-groups from it. */
    ch_range channel_range(size_t channel, size_t recv_count) const {
        const size_t unit_beg = num_align_units * channel / num_channels;
        const size_t unit_end = num_align_units * (channel + 1) / num_channels;
        return { std::min(recv_count, unit_beg * align_elems), std::min(recv_count, unit_end * align_elems) };
    }

    // The longest channel, hence the largest data_count any chunk will see. A max over the ranges
    // rather than a formula, the ragged tail being clamped per channel.
    size_t max_channel_count(size_t recv_count) const {
        size_t count_max = 0;
        for (size_t channel = 0; channel < num_channels; channel++) {
            count_max = std::max(count_max, channel_range(channel, recv_count).count());
        }
        return count_max;
    }
};

inline reduce_scatter_device_proxy_geometry reduce_scatter_device_proxy_make_geometry(size_t recv_count,
                                                                                      size_t dsize,
                                                                                      size_t staging_slot_size) {
    const size_t align_bytes = reduce_scatter_device_proxy_align_bytes();
    reduce_scatter_device_proxy_geometry geom;

    // A knob set below a dtype size costs alignment but not correctness - the scalar tail of
    // copy_at()/reduce_at() covers any count.
    geom.align_elems = std::max<size_t>(1, align_bytes / dsize);
    geom.num_align_units = (recv_count + geom.align_elems - 1) / geom.align_elems;

    /* Three rank-invariant caps: the work floor, one alignment unit of the output block per
     * channel so none starts out empty, and one of every staging slot per channel so chunk_size
     * cannot round to zero. */
    geom.num_channels = std::min(reduce_scatter_device_proxy_num_channels,
                                 recv_count * dsize / reduce_scatter_device_proxy_min_bytes_per_channel);
    geom.num_channels = std::min(geom.num_channels, geom.num_align_units);
    geom.num_channels = std::min(geom.num_channels, staging_slot_size / align_bytes);
    geom.num_channels = std::max<size_t>(1, geom.num_channels);

    // A channel's fixed share of every staging slot, hence its per-chunk ceiling. Rounded down to
    // align_bytes, so every chunk but a channel's last starts and ends on an alignment unit;
    // non-zero, the cap above leaving chunk_size >= align_bytes.
    const size_t chunk_size = staging_slot_size / geom.num_channels / align_bytes * align_bytes;
    geom.chunk_count = chunk_size / dsize;

    return geom;
}

// Gate for the flat-ring path, kept out of the per-dtype template so every rejection is logged
// from one place as can_use_sycl_kernels() does. Predicate only; make_topology() builds the ring.
inline bool reduce_scatter_device_proxy_can_use(ccl::reduction reduction,
                                                ccl_comm *global_comm,
                                                std::shared_ptr<ccl_comm> p2p_comm) {
    /* The proxy service only boots when a collective asks for it (init_proxy_service()), so require
     * our own flag rather than reading is_started() alone: another collective's flag having started
     * the service is not a request to route reduce_scatter through it. */
    if (!ccl::global_data::env().sycl_enable_reduce_scatter_scaleout_proxy) {
        LOG_DEBUG("reduce_scatter device proxy: ", CCL_SYCL_ENABLE_REDUCE_SCATTER_SCALEOUT_PROXY, " is not set");
        return false;
    }

    ccl_proxy_service *proxy = ccl::global_data::get().proxy_service.get();
    if (!proxy || !proxy->is_started()) {
        LOG_DEBUG("reduce_scatter device proxy: proxy service is not started");
        return false;
    }

    // RDMA-only by design: the proxy thread posts ATL send/recv straight on the device pointers
    // the kernel gives it, with no host bounce buffer.
    if (!proxy->staging().m_is_rdma_supported) {
        LOG_DEBUG("reduce_scatter device proxy: GPU RDMA is not supported");
        return false;
    }

    // Intra-domain hops are IPC writes into peers' large tmp buffers, and the same flag decides
    // whether those buffers were allocated on the device at all (sycl_coll_base.cpp).
    if (!global_comm->get_node_comm()->get_topo_manager().has_p2p_access()) {
        LOG_DEBUG("reduce_scatter device proxy: no p2p access within the node");
        return false;
    }

    /* Nothing to send off-domain: every hop would be an IPC write, which the scale-up kernels do
     * better. Comm-uniform like every other condition here, domain membership being symmetric - if
     * one rank's domain spans the comm then every rank's does, so no rank picks this path while
     * another falls back. Covers world == 1 too. */
    if (p2p_comm->size() == global_comm->size()) {
        LOG_DEBUG("reduce_scatter device proxy: the p2p domain spans the whole comm");
        return false;
    }

    // coll_initExt() still allocates the narrow one-cell-per-peer counter region, so
    // p2p_barrier_pair_wg() on any channel but 0 would write past it.
    if (global_comm->is_multi_thread_instance()) {
        LOG_DEBUG("reduce_scatter device proxy: multi-threaded instance has narrow "
                  "pairwise-barrier counters");
        return false;
    }

    // Same gap on coll_init()'s UMF branch: the IPC-visible flag cells are sized for
    // MAX_P2P_CHANNELS there, but the local counters are not.
    if (ccl::global_data::env().umf_enable) {
        LOG_DEBUG("reduce_scatter device proxy: UMF allocation path has narrow "
                  "pairwise-barrier counters");
        return false;
    }

    // avg is handled by the sycl_average() post-pass; a custom (pre_mul_sum) reduction needs the
    // staging that reduce_scatter_sycl_multi_domain does.
    if (ccl_reduction_type_storage::is_custom(reduction)) {
        LOG_DEBUG("reduce_scatter device proxy: custom reduction is not supported");
        return false;
    }

    // Both ends of CCL_SYCL_TMP_BUF_SIZE: a request carries byte counts as uint32_t, so an
    // oversized value truncates, and one rounding the slot size to zero divides by zero later.
    const size_t staging_slot_size = reduce_scatter_device_proxy_staging_slot_size();
    if (staging_slot_size == 0 || staging_slot_size > static_cast<size_t>(UINT32_MAX)) {
        LOG_DEBUG("reduce_scatter device proxy: tmp staging slot size ",
                  staging_slot_size,
                  " (",
                  reduce_scatter_device_proxy_staging_slots,
                  " slots over ",
                  ccl::global_data::env().sycl_tmp_buf_size,
                  " bytes of tmp buffer) is unusable as a request byte count");
        return false;
    }

    // No rank-layout condition left: each rank resolves its two hops by lookup, so any domain sizes
    // and any interleaving work, a domain of one included - there both hops are remote and the rank
    // has a send and a recv outstanding at once, which the host consumer's per-direction cursors
    // handle (request_ring_host.hpp).
    LOG_DEBUG("reduce_scatter selects device proxy flat ring: world ",
              global_comm->size(),
              ", rank ",
              global_comm->rank(),
              ", p2p domain ",
              p2p_comm->size(),
              ", ",
              reduce_scatter_device_proxy_staging_slots,
              " staging slots of ",
              staging_slot_size,
              " bytes");

    return true;
}

/* p2p_comm rank of a global_comm rank, or -1 when that rank sits outside our p2p domain, which is
 * what picks each hop's transport. One lookup: a subcomm's rank map is in its *parent's* rank
 * space, and p2p_comm's parent is global_comm, so `rank` is already in it. Lifting it to the world first -
 * the idiom of ccl_comm::get_node_rank() - is only right between sibling subcomms, and breaks on a
 * split global_comm. No layout assumption is left in here, so domains of any size, shape or
 * interleaving resolve, a domain of one giving -1 both ways. The two hops agree pairwise for free:
 * domain membership is symmetric, so our right hop being remote is exactly that neighbour's left
 * hop being remote. */
inline int reduce_scatter_device_proxy_local_peer(ccl_comm *p2p_comm, int rank) {
    if (!p2p_comm->try_get_rank_from_global(rank)) {
        return -1;
    }
    return p2p_comm->get_rank_from_global(rank);
}

/* Resolves the ring, once the gate above has said yes. */
inline reduce_scatter_device_proxy_topology reduce_scatter_device_proxy_make_topology(
    ccl_comm *global_comm,
    std::shared_ptr<ccl_comm> p2p_comm) {
    ccl_proxy_service *proxy = ccl::global_data::get().proxy_service.get();
    const int N = global_comm->size();
    const int rank = global_comm->rank();

    reduce_scatter_device_proxy_topology topo;
    topo.ring_size = N;
    topo.ring_rank = rank;
    topo.left_rank = (rank - 1 + N) % N;
    topo.right_rank = (rank + 1) % N;
    // Per hop and independently: either may be local or remote, including both remote.
    topo.local_left_peer = reduce_scatter_device_proxy_local_peer(p2p_comm.get(), topo.left_rank);
    topo.local_right_peer = reduce_scatter_device_proxy_local_peer(p2p_comm.get(), topo.right_rank);

    // Remote peers are named by global_comm rank, so that is the comm the proxy thread resolves them
    // against. Held by value in the params, so global_comm need only outlive the call.
    topo.ring = proxy->device_ring_handle();
    topo.comm = ccl_make_device_comm_proxy(global_comm);

    LOG_DEBUG("reduce_scatter device proxy ring: left ",
              topo.left_rank,
              (topo.local_left_peer >= 0 ? " (local)" : " (remote)"),
              ", right ",
              topo.right_rank,
              (topo.local_right_peer >= 0 ? " (local)" : " (remote)"));

    return topo;
}

struct reduce_scatter_device_proxy_params {
    // Drives the per-delivery pairwise handshakes. No barrier_data: the entry rendezvous is a
    // separate launch, comm_barrier() not spanning work-groups (see the driver).
    ccl_comm_pair_flag_data pflag = { -1, -1 };
    reduce_scatter_device_proxy_geometry geometry{};
    // The tmp region as base + slot size, the slots being contiguous. remote_work_base is the
    // right neighbour's base, null when that hop is remote - remote_staging_slot() is then unused.
    void *work_base = nullptr;
    void *remote_work_base = nullptr;
    size_t staging_slot_size = 0;
    reduce_scatter_device_proxy_topology topo{};

    // The region of sycl_tmp_buf holding ring delivery `delivery`, at byte offset chan_off inside it
    // - the caller's ch_staging_offset, so two channels never name the same bytes.
    void *staging_slot(int delivery, size_t chan_off = 0) const {
        return staging_slot_in(work_base, delivery, chan_off);
    }
    void *remote_staging_slot(int delivery, size_t chan_off = 0) const {
        return staging_slot_in(remote_work_base, delivery, chan_off);
    }

private:
    void *staging_slot_in(void *base, int delivery, size_t chan_off) const {
        const int slot = delivery % reduce_scatter_device_proxy_staging_slots;
        return (char *)base + slot * staging_slot_size + chan_off;
    }
};

/* One ring delivery, as three calls: post_recv() for the arrival from the left, then whatever the
 * group computes and hands on, then post_send() to the right, then sync() to close the delivery.
 * The two hops are independent - either may stay in the p2p domain or leave it, both remote being
 * the one-rank-per-domain case - so each of the three does its own half and does nothing when that
 * hop is local. Publishing no request on a delivery is fine; claiming a ring sequence without
 * publishing it is not, a hole wedging the host's ring reader - every direction - forever.
 *
 * The local half is a pairwise barrier, not the ring p2p_barrier, which on a domain-boundary rank
 * would signal p2p rank (p+1) % P - the wrong partner once the real right neighbour is remote. Its
 * per-(channel, peer) counters also keep the two calls in sync() from interfering with each other,
 * with sibling channels, or with ring barriers elsewhere. Bidirectional on purpose: it lock-steps the
 * peer onto the same delivery, bounding drift at one, where a one-way ring barrier would allow N-1 -
 * a whole chunk's slots overwritten under the peer. A rank holding both halves chains them
 * right-then-left; a rank holding neither runs no barrier at all and is paced by its two transfers.
 *
 * The remote requests are posted before the local handshakes and waited on after, so they overlap
 * them - and the recv is posted before the group computes what the send carries, so it overlaps the
 * reduce too. That costs nothing in slot safety: a delivery receives one slot ahead of every slot
 * its own compute touches (see the call sites), so the NIC never writes what the group is still
 * reading, and each of our slots still has exactly one writer - the NIC when the left hop is remote,
 * the left peer's push-copy when it is local.
 *
 * Work-group scoped throughout, `channel` being this work-group's index. The request ring is the
 * only state shared with siblings - safe because their byte ranges were fixed before the chunk loop,
 * and because one ATL tag per (comm, channel) matches a send with the same channel's recv on the
 * next rank however the two ranks raced (see request_ring_host.hpp). */

// Take delivery of `dst` from the left neighbour, when that hop is remote. Needs neither barrier nor
// fence: nothing this group wrote has to be visible for the NIC to fill `dst`, so the leader posts it
// and the group walks straight into its compute. The sequence is leader-only, and 0 when the hop is
// local - the convention of the submit wrappers; sync() consumes it.
static inline uint32_t reduce_scatter_device_proxy_post_recv(const reduce_scatter_device_proxy_params &params,
                                                             const sycl::nd_item<1> it,
                                                             int channel,
                                                             void *dst,
                                                             uint32_t bytes) {
    if (params.topo.local_left_peer >= 0) {
        return 0;
    }
    return request_ring_submit_recv_group(it.get_group(),
                                          params.topo.ring,
                                          dst,
                                          bytes,
                                          params.topo.left_rank,
                                          params.topo.comm,
                                          static_cast<uint32_t>(channel));
}

// Hand `src` to the right neighbour, when that hop is remote. `src` was just written by the whole
// group, so the group barrier orders those stores and the system-scope release makes them visible
// off-device - the NIC reads this buffer - before the request is published.
static inline uint32_t reduce_scatter_device_proxy_post_send(const reduce_scatter_device_proxy_params &params,
                                                             const sycl::nd_item<1> it,
                                                             int channel,
                                                             void *src,
                                                             uint32_t bytes) {
    if (params.topo.local_right_peer >= 0) {
        return 0;
    }

    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

    return request_ring_submit_send_group(it.get_group(),
                                          params.topo.ring,
                                          src,
                                          bytes,
                                          params.topo.right_rank,
                                          params.topo.comm,
                                          static_cast<uint32_t>(channel));
}

// Close the delivery: the pairwise handshake for each hop that stayed in the p2p domain, then the
// wait for each one that went over the network. Locality is re-derived here rather than passed, so
// the waits cannot disagree with what the two posts above submitted.
static inline void reduce_scatter_device_proxy_sync(const reduce_scatter_device_proxy_params &params,
                                                    const sycl::nd_item<1> it,
                                                    int channel,
                                                    uint32_t recv_seq,
                                                    uint32_t send_seq) {
    const bool right_is_local = params.topo.local_right_peer >= 0;
    const bool left_is_local = params.topo.local_left_peer >= 0;

    if (right_is_local) {
        p2p_barrier_pair_wg(params.pflag, params.topo.local_right_peer, channel, it);
    }
    if (left_is_local) {
        p2p_barrier_pair_wg(params.pflag, params.topo.local_left_peer, channel, it);
    }

    // Each wait releases the group with an acquire that orders the NIC-written payload before its
    // loads. A rank with both hops remote pays the group barrier of each - noise beside a transfer.
    if (!left_is_local) {
        request_ring_wait_group(it.get_group(), params.topo.ring, recv_seq);
    }
    if (!right_is_local) {
        request_ring_wait_group(it.get_group(), params.topo.ring, send_seq);
    }
}

template <typename T, int vec_size>
void inline reduce_scatter_device_proxy_kernel_wrap(const sycl::nd_item<1> it,
                                                    const void *send_buf,
                                                    void *recv_buf,
                                                    size_t recv_count,
                                                    ccl_reduction_data reduction,
                                                    const reduce_scatter_device_proxy_params params) {
    // Local, not global: this work-group *is* the channel, so its element range is indexed by the
    // group-local id - hence copy_at / reduce_at rather than copy / reduce.
    const size_t num_threads = it.get_local_range(0);
    const size_t lid = it.get_local_linear_id();
    const int channel = static_cast<int>(it.get_group_linear_id());

    const int N = params.topo.ring_size;
    const bool right_is_local = params.topo.local_right_peer >= 0;

    const int dsize = sizeof(T);
    const size_t recv_bytes = recv_count * dsize;

    // This channel's element range of this rank's output block, computed once and never changed:
    // channels are unsynchronised, so their drift is unbounded and fixed ranges are the only thing
    // keeping them off each other's memory.
    const auto ch = params.geometry.channel_range(channel, recv_count);
    const size_t ch_count = ch.count();

    const size_t chunk_count = params.geometry.chunk_count;
    const size_t ch_staging_offset = channel * chunk_count * dsize;
    // This channel's own trip count, not the longest channel's: only channel c on the next rank has
    // to publish as many requests as channel c here, so a short channel just runs one trip fewer.
    const size_t num_chunks = (ch_count + chunk_count - 1) / chunk_count;

    // The chunk loop needs no barrier of its own: a chunk's last step runs no delivery, hence no
    // pairwise barrier, so a rank may enter the next chunk one delivery ahead of its peer - which is
    // what the fourth staging slot is reserved for. The entry rendezvous is in the driver.

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        // Every trip moves something, num_chunks being this channel's own ceiling; only the last
        // one can be partial.
        const size_t ch_done = chunk_idx * chunk_count;
        const size_t chunk_beg = ch.beg + ch_done;
        const size_t data_count = std::min(chunk_count, ch_count - ch_done);
        const size_t req_threads = data_count / vec_size + data_count % vec_size;
        const uint32_t delivery_bytes = static_cast<uint32_t>(data_count * dsize);
        const size_t chunk_offset = chunk_beg * dsize;

        // Deliveries continue where the previous chunk's left off rather than restarting at 0: a
        // per-chunk index would make the boundary a backwards jump in the slot sequence, and the
        // reuse argument would hinge on ((N - 2) % staging slots) != 0.
        const int delivery_base = static_cast<int>(chunk_idx) * (N - 1);

        // Walks one block back per hop: delivery 0 forwards block rank-1, the last this rank's own.
        int block_idx = (params.topo.ring_rank - 1 + N) % N;

        // Delivery 0 is the raw block, straight out of the user's send buffer.
        {
            void *src = (char *)send_buf + block_idx * recv_bytes + chunk_offset;
            void *dst = params.staging_slot(delivery_base, ch_staging_offset);

            // Posted first so the arrival runs while the group pushes: `dst` is this rank's slot,
            // which the push below never touches - it writes the right peer's.
            const uint32_t recv_seq =
                reduce_scatter_device_proxy_post_recv(params, it, channel, dst, delivery_bytes);

            if (right_is_local) {
                void *out2 = params.remote_staging_slot(delivery_base, ch_staging_offset);
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    copy_at<T, vec_size>(out2, src, data_count, lid + offset);
                }
            }

            const uint32_t send_seq =
                reduce_scatter_device_proxy_post_send(params, it, channel, src, delivery_bytes);
            reduce_scatter_device_proxy_sync(params, it, channel, recv_seq, send_seq);
        }

        // Iteration i takes the arrival in delivery i's slot, adds this rank's copy of the same
        // block on top in place, and sends the result on as delivery i + 1. Only N - 2 of the N - 1
        // arrivals are forwarded, the last being peeled out below.
        for (int i = 0; i < N - 2; i++) {
            block_idx = (block_idx - 1 + N) % N;

            // In place: this slot is both the reduce's out and its in2.
            void *acc = params.staging_slot(delivery_base + i, ch_staging_offset);
            void *in1 = (char *)send_buf + block_idx * recv_bytes + chunk_offset;
            void *next = params.staging_slot(delivery_base + i + 1, ch_staging_offset);

            // Posted first so the arrival runs while the group reduces: `next` is one slot ahead of
            // `acc`, and the push below writes the right peer's slot, not ours.
            const uint32_t recv_seq =
                reduce_scatter_device_proxy_post_recv(params, it, channel, next, delivery_bytes);

            if (right_is_local) {
                // Each item pushes the element it just reduced.
                void *out2 = params.remote_staging_slot(delivery_base + i + 1, ch_staging_offset);
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce_copy_at<T, vec_size>(acc, in1, acc, out2, data_count, lid + offset, reduction);
                }
            }
            else {
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce_at<T, vec_size>(acc, in1, acc, data_count, lid + offset, reduction);
                }
            }

            const uint32_t send_seq =
                reduce_scatter_device_proxy_post_send(params, it, channel, acc, delivery_bytes);
            reduce_scatter_device_proxy_sync(params, it, channel, recv_seq, send_seq);
        }

        // The last arrival is reduced straight into the user's recv buffer and forwarded nowhere,
        // so it needs neither a push nor a delivery.
        block_idx = (block_idx - 1 + N) % N;
        {
            void *out = (char *)recv_buf + chunk_offset;
            void *in1 = (char *)send_buf + block_idx * recv_bytes + chunk_offset;
            void *in2 = params.staging_slot(delivery_base + N - 2, ch_staging_offset);
            for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                reduce_at<T, vec_size>(out, in1, in2, data_count, lid + offset, reduction);
            }
        }
    }
}

template <typename T>
sycl::event reduce_scatter_device_proxy_ring(const void *send_buf,
                                             void *recv_buf,
                                             size_t recv_count,
                                             ccl::datatype dtype,
                                             ccl::reduction reduction,
                                             ccl_comm *global_comm,
                                             std::shared_ptr<ccl_comm> p2p_comm,
                                             ccl_stream *global_stream,
                                             const reduce_scatter_device_proxy_topology &topo,
                                             const ccl::vector_class<ccl::event> &deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, recv_count, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    // Index 0 is the base of the whole tmp allocation on both sides; the slot arithmetic lives in
    // reduce_scatter_device_proxy_params::staging_slot().
    const size_t staging_slot_size = reduce_scatter_device_proxy_staging_slot_size();
    void *work_base = get_large_tmp_buf(0, global_comm);
    void *remote_work_base = nullptr;
    // Only when the right hop stays in the domain, so no out-of-range peer index is formed.
    if (topo.local_right_peer >= 0) {
        remote_work_base = get_remote_large_tmp_buf(0, p2p_comm.get())[topo.local_right_peer];
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

    // One work-group per channel, sized here because the size is not rank-invariant (it depends on
    // vec_size and max_work_group_size) and affects nothing but local indexing: the largest legal
    // one rather than CCL_SYCL_WORK_GROUP_SIZE, capped at what one chunk can keep busy.
    const auto geom = reduce_scatter_device_proxy_make_geometry(recv_count, dsize, staging_slot_size);
    const size_t count_max = geom.max_channel_count(recv_count);
    const size_t max_chunk_count = std::min(geom.chunk_count, count_max);
    const size_t kernel_threads_actual = max_chunk_count / vec_size + max_chunk_count % vec_size;

    constexpr size_t sg_size = 16; // must match the reqd_sub_group_size on the kernel
    const size_t wg_size = std::max(sg_size,
                                    std::min(((kernel_threads_actual + sg_size - 1) / sg_size) * sg_size,
                                             (max_work_group_size / sg_size) * sg_size));
    sycl::nd_range<1> nd_range(geom.num_channels * wg_size, wg_size);

    // The longest channel's trip count; the kernel derives each channel's own.
    const size_t num_chunks = std::max<size_t>(1, (count_max + geom.chunk_count - 1) / geom.chunk_count);
    LOG_DEBUG("reduce_scatter device proxy flat ring: ",
              geom.num_channels,
              " channels of ",
              wg_size,
              " threads, ",
              num_chunks,
              " chunks of up to ",
              geom.chunk_count * dsize,
              " bytes per channel, recv_count ",
              recv_count);

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    reduce_scatter_device_proxy_params params;
    params.pflag = p2p_comm->pair_flag_data();
    params.geometry = geom;
    params.work_base = work_base;
    params.remote_work_base = remote_work_base;
    params.staging_slot_size = staging_slot_size;
    params.topo = topo;

    // Entry barrier: keeps this op's staging writes off slots a previous op may still be reading.
    // Over node_comm, not p2p_comm - the large tmp buffer it protects is node-scoped.
    // TODO: add/implement real scale-out barrier on global_comm ranks
    invoke_barrier(global_comm->get_node_comm(), q, dep_events, false /* use_cpu */);

    if (use_full_vector) {
        work_event = q.parallel_for(nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
            reduce_scatter_device_proxy_kernel_wrap<T, full_vec_size>(
                it, send_buf, recv_buf, recv_count, reduction_op, params);
        });
    }
    else {
        work_event = q.parallel_for(nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
            reduce_scatter_device_proxy_kernel_wrap<T, partial_vec_size>(
                it, send_buf, recv_buf, recv_count, reduction_op, params);
        });
    }

    if (reduction == ccl::reduction::avg) {
        // The kernel only sums, and the flat ring accumulated over every rank rather than the local
        // domain, so divide by the full world size.
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, recv_count, global_comm->size(), dtype, evs);
    }

    return work_event;
}

#ifndef CCL_NO_EXTERN_TEMPLATE_INST
// Suppress local instantiation in TUs that only call this impl via invoke_collective; the
// instantiations live in the sibling inst/ .cpp files.

#define TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(DT) \
    template sycl::event reduce_scatter_device_proxy_ring<DT>(const void *send_buf, \
                                                              void *recv_buf, \
                                                              size_t recv_count, \
                                                              ccl::datatype dtype, \
                                                              ccl::reduction reduction, \
                                                              ccl_comm *global_comm, \
                                                              std::shared_ptr<ccl_comm> p2p_comm, \
                                                              ccl_stream *global_stream, \
                                                              const reduce_scatter_device_proxy_topology &topo, \
                                                              const ccl::vector_class<ccl::event> &deps)

extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(short);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(int8_t);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(uint8_t);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(int);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(uint32_t);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(int64_t);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(uint64_t);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(float);
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(double);
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(sycl::half);
#endif
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
extern TEMPLATE_REDUCE_SCATTER_DEVICE_PROXY_RING(sycl::ext::oneapi::bfloat16);
#endif
#endif // !CCL_NO_EXTERN_TEMPLATE_INST
