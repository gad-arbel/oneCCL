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

#include <cstddef>

#include "common/global/global.hpp"
#include "comm/comm.hpp"
#include "common/stream/stream.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

// CE/EU pipeline (copy-engine copies overlapped with EU reductions): plan and chunking.
// The schedules are in reduce_scatter_large_sycl_ring.hpp.

// measured on CRI
inline constexpr size_t CE_EU_ALLREDUCE_CHUNK_BYTES = 8 * 1024 * 1024;
inline constexpr size_t CE_EU_REDUCE_SCATTER_CHUNK_BYTES = 4 * 1024 * 1024;
inline constexpr size_t CE_EU_AUTO_MIN_CHUNK_BYTES = 2 * 1024 * 1024;
inline constexpr size_t CE_EU_AUTO_MIN_CHUNKS = 4;
// below this, the per-chunk barrier and kernel launch cost more than the copy
inline constexpr size_t CE_EU_MIN_CHUNK_BYTES = 64 * 1024;
// more slots add no bandwidth on CRI
inline constexpr size_t CE_EU_TWO_RANK_SLOTS = 16;
inline constexpr size_t CE_EU_MAX_SLOTS = ARC_MAX_NUM;
inline constexpr size_t CE_EU_MIN_SLOTS = 2;
inline constexpr size_t CE_EU_MIN_CHUNKS = 2;
inline constexpr int CE_EU_MIN_RANKS = 2;
inline constexpr int CE_EU_MIN_RING_RANKS = 3; // fewer ranks use the 2-rank schedule
inline constexpr int CE_EU_MAX_RANKS = 5;
// in the ring, the first push of a chunk waits for the barrier this many time slots back
inline constexpr size_t CE_EU_FIRST_PUSH_LAG = 2;
inline constexpr size_t CE_EU_VECTOR_BYTES = 16;
inline constexpr size_t CE_EU_NARROW_VECTOR_BYTES = 4;
inline constexpr int CE_EU_SUB_GROUP_SIZE = 16;

struct ce_eu_pipeline_plan_t {
    bool use_pipeline = false;
    // null: the copies run on the stream queue (same schedule, no overlap), see ce_note
    const sycl::queue *ce_queue_ptr = nullptr;
    int ce_ordinal = -1;
    int ce_index = -1;
    const char *ce_note = nullptr;
    size_t slots = 0; // tmp-buffer slots
    size_t chunk_bytes = 0; // also the slot size
    size_t elems_per_chunk = 0;
    size_t num_chunks = 0;

    sycl::queue ce_queue(const sycl::queue &stream_queue) const {
        return ce_queue_ptr ? *ce_queue_ptr : stream_queue;
    }
};

// All values that enter the decision must be equal on all ranks.
struct ce_eu_pipeline_request_t {
    const char *collective = "";
    ccl_stream *stream = nullptr;
    ccl_comm *comm = nullptr;
    int ranks = 0; // size of the scale-up communicator
    bool is_cpu_barrier = false;
    bool has_whole_vectors = true; // is_whole_vector_count()
    size_t dsize = 0;
    size_t message_bytes = 0;
    size_t count_per_rank = 0; // elements each rank reduces
    size_t preferred_chunk_bytes = 0; // automatic chunk size for large messages
};

// Pipeline or serial path, chunks and slots. The two paths use different numbers of barriers,
// so the decision uses only rank-invariant values (no buffer addresses), and a missing copy
// engine queue does not change it. Logs the decision at info level when it changes.
// ce_queue_index (copies) and compute_queue_index (the stream queue, or unknown) are
// <engine group>.<engine>; selected_path=serial is the path without the
// pipeline, which may still use the copy engine.
ce_eu_pipeline_plan_t plan_ce_eu_pipeline(const ce_eu_pipeline_request_t &r);

// The chunks of a rank's part: all of one size but the last.
struct ce_eu_chunking_t {
    ce_eu_chunking_t(const ce_eu_pipeline_plan_t &plan, size_t count_per_rank, size_t dsize_in)
            : elems_per_chunk(plan.elems_per_chunk),
              full_chunks(count_per_rank / plan.elems_per_chunk),
              tail_elems(count_per_rank % plan.elems_per_chunk),
              dsize(dsize_in) {}

    size_t offset_bytes(size_t c) const {
        return c * elems_per_chunk * dsize;
    }
    size_t elems(size_t c) const {
        return c < full_chunks ? elems_per_chunk : tail_elems;
    }

    size_t elems_per_chunk;
    size_t full_chunks;
    size_t tail_elems;
    size_t dsize;
};

// The kernels take the wide vector path. Counts and env only: addresses differ between ranks.
inline bool is_whole_vector_count(size_t count, size_t dsize) {
    return dsize >= CE_EU_NARROW_VECTOR_BYTES || (ccl::global_data::env().sycl_full_vector &&
                                                  (count * dsize) % CE_EU_NARROW_VECTOR_BYTES == 0);
}

// Halves the preferred chunk, down to CE_EU_AUTO_MIN_CHUNK_BYTES, until a rank's part has
// CE_EU_AUTO_MIN_CHUNKS chunks.
inline size_t auto_pipeline_chunk_bytes(size_t preferred_bytes, size_t part_bytes) {
    size_t chunk = preferred_bytes;
    while (chunk > CE_EU_AUTO_MIN_CHUNK_BYTES && part_bytes / chunk < CE_EU_AUTO_MIN_CHUNKS) {
        chunk /= 2;
    }
    return chunk;
}

// Slot reuse in a ring of N ranks: a push in time slot t overwrites a slot last used in time
// slot t'. The barriers prove the neighbour is done with it when t - t' >= N + lag - 1, where
// lag is how many time slots back the barrier the push waits for is.
constexpr bool is_ring_slot_count_safe(int N, size_t slots) {
    const size_t steps = N - 1;
    for (size_t i = 0; i < steps; i++) {
        const size_t c = slots; // a chunk whose slot had an earlier user
        const size_t previous = c * steps + i - slots;
        const size_t t = c + i;
        const size_t t_previous = previous / steps + previous % steps;
        const size_t lag = i == 0 ? CE_EU_FIRST_PUSH_LAG : 1;
        if (t - t_previous < static_cast<size_t>(N) + lag - 1) {
            return false;
        }
    }
    return true;
}

constexpr bool has_safe_ring_slot_count(int N, size_t max_slots) {
    for (size_t slots = CE_EU_MIN_SLOTS; slots <= max_slots; slots++) {
        if (is_ring_slot_count_safe(N, slots)) {
            return true;
        }
    }
    return false;
}

// N ranks need N * N - 1 slots, so more than CE_EU_MAX_RANKS do not fit in CE_EU_MAX_SLOTS.
constexpr bool is_ring_slot_rule_valid() {
    for (int n = CE_EU_MIN_RING_RANKS; n <= CE_EU_MAX_RANKS; n++) {
        const size_t needed = static_cast<size_t>(n * n - 1);
        if (!is_ring_slot_count_safe(n, needed) || is_ring_slot_count_safe(n, needed - 1)) {
            return false;
        }
    }
    return !has_safe_ring_slot_count(CE_EU_MAX_RANKS + 1, CE_EU_MAX_SLOTS);
}
static_assert(is_ring_slot_rule_valid(), "the ring slot rule does not match CE_EU_MAX_RANKS");
