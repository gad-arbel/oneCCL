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

#include "coll/algorithms/utils/sycl_ce_eu_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "coll/algorithms/utils/barrier_data.hpp"
#include "coll/selection/policy/policy_env.hpp"
#include "common/api_wrapper/ze_api_wrapper.hpp"

// <engine group>.<engine> of the stream queue, when it is an immediate command list and the
// driver can tell; "unknown" otherwise
static std::string stream_queue_index(ccl_stream *stream) {
    const sycl::queue q = stream->get_native_stream();
    if (q.get_device().get_backend() != sycl::backend::ext_oneapi_level_zero ||
        !zeCommandListGetOrdinal || !zeCommandListImmediateGetIndex) {
        return "unknown";
    }
    const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
    const ze_command_list_handle_t *list_ptr = std::get_if<ze_command_list_handle_t>(&native);
    uint32_t ordinal = 0;
    uint32_t index = 0;
    if (!list_ptr || zeCommandListGetOrdinal(*list_ptr, &ordinal) != ZE_RESULT_SUCCESS ||
        zeCommandListImmediateGetIndex(*list_ptr, &index) != ZE_RESULT_SUCCESS) {
        return "unknown";
    }
    return std::to_string(ordinal) + "." + std::to_string(index);
}

ce_eu_pipeline_plan_t plan_ce_eu_pipeline(const ce_eu_pipeline_request_t &r) {
    ce_eu_pipeline_plan_t plan;
    const auto &env = ccl::global_data::env();
    if (!env.enable_sycl_ce_eu_pipeline) {
        return plan;
    }

    size_t chunk_bytes =
        auto_pipeline_chunk_bytes(r.preferred_chunk_bytes, r.count_per_rank * r.dsize);
    if (env.sycl_ce_eu_pipeline_chunk_size) {
        CCL_THROW_IF_NOT(env.sycl_ce_eu_pipeline_chunk_size >= CE_EU_MIN_CHUNK_BYTES,
                         CCL_SYCL_CE_EU_PIPELINE_CHUNK_SIZE,
                         " = ",
                         env.sycl_ce_eu_pipeline_chunk_size,
                         " is too small, the minimum is ",
                         CE_EU_MIN_CHUNK_BYTES,
                         " bytes");
        // at most a CE_EU_MIN_SLOTS-th of the tmp buffer, with a warning
        chunk_bytes = get_simple_ring_segment_size(CCL_SYCL_CE_EU_PIPELINE_CHUNK_SIZE,
                                                   env.sycl_ce_eu_pipeline_chunk_size,
                                                   static_cast<int>(CE_EU_MIN_SLOTS),
                                                   large_tmp_buf_align);
    }
    chunk_bytes = std::min(chunk_bytes, env.sycl_tmp_buf_size / CE_EU_MIN_SLOTS);
    chunk_bytes = chunk_bytes / large_tmp_buf_align * large_tmp_buf_align;
    plan.chunk_bytes = chunk_bytes;
    plan.elems_per_chunk = chunk_bytes / r.dsize;
    plan.num_chunks = plan.elems_per_chunk
                          ? (r.count_per_rank + plan.elems_per_chunk - 1) / plan.elems_per_chunk
                          : 0;
    plan.slots =
        chunk_bytes ? std::min(CE_EU_TWO_RANK_SLOTS, env.sycl_tmp_buf_size / chunk_bytes) : 0;
    if (r.ranks >= CE_EU_MIN_RING_RANKS && chunk_bytes) {
        // the most safe slots that fit; the safe counts have gaps, so search downward
        plan.slots = std::min(CE_EU_MAX_SLOTS, env.sycl_tmp_buf_size / chunk_bytes);
        while (plan.slots >= CE_EU_MIN_SLOTS && !is_ring_slot_count_safe(r.ranks, plan.slots)) {
            plan.slots--;
        }
    }

    const char *reason = nullptr;
    if (!ccl::selection::call_env().sycl_copy_engine) {
        reason = "copy_engine_off";
        LOG_WARN_ONCE("CCL_SYCL_CE_EU_PIPELINE=1 has no effect without CCL_SYCL_COPY_ENGINE=1; ");
    }
    else if (r.ranks < CE_EU_MIN_RANKS) {
        reason = "too_few_ranks";
    }
    else if (r.is_cpu_barrier) {
        reason = "cpu_barrier";
    }
    else if (use_recording_path(r.stream)) {
        reason = "recording_path";
    }
    else if (r.comm->is_multi_thread_instance() ||
             ccl::global_data::get().shared_data->is_multi_thread_instance) {
        // cross-queue waits can hang with multi-thread communicators
        reason = "multi_thread";
    }
    else if (!r.has_whole_vectors) {
        // the narrow vector path must not be split (see chunks_use_full_vector)
        reason = "narrow_vector";
    }
    else if (chunk_bytes < CE_EU_MIN_CHUNK_BYTES) {
        reason = "small_tmp_buf";
    }
    else if (plan.slots < CE_EU_MIN_SLOTS) {
        reason = "too_few_slots";
    }
    else if (plan.num_chunks < CE_EU_MIN_CHUNKS) {
        reason = "single_chunk";
    }
    else {
        // not the first engine: on CRI the stream queue's copies run there and can block ours
        // (DEVICE_LOST); BMG has only one
        const pinned_copy_queue &pq = get_stream_copy_engine_queue(
            r.stream, r.stream->ce_pipeline_queue, copy_engine_pick::last_or_only);
        if (pq.queue) {
            plan.ce_queue_ptr = &*pq.queue;
            plan.ce_ordinal = pq.ordinal;
            plan.ce_index = pq.index;
        }
        else {
            plan.ce_note = pq.fallback_reason;
            // LOG_WARN_ONCE prints on the root rank only
            static std::atomic<bool> is_warned{ false };
            if (!is_warned.exchange(true)) {
                LOG_WARN("CCL_SYCL_CE_EU_PIPELINE: no copy engine queue (",
                         pq.fallback_reason,
                         "), the copies run on the stream queue and do not overlap the "
                         "reductions");
            }
        }
    }
    plan.use_pipeline = (reason == nullptr);

    if (ccl_logger::get_instance().get_log_level() >= ccl_log_level::info) {
        // when the size or the outcome changes
        struct logged_plan_t {
            size_t message_bytes = SIZE_MAX;
            const char *reason = nullptr;
            const char *ce_note = nullptr;
        };
        static thread_local std::unordered_map<std::string, logged_plan_t> last_logged;
        logged_plan_t &last = last_logged[std::string(r.collective) + ":" +
                                          std::to_string(r.ranks) + ":" + std::to_string(r.dsize)];
        if (last.message_bytes != r.message_bytes || last.reason != reason ||
            last.ce_note != plan.ce_note) {
            last = { r.message_bytes, reason, plan.ce_note };
            if (plan.use_pipeline) {
                LOG_INFO("collective=",
                         r.collective,
                         " ranks=",
                         r.ranks,
                         " selected_path=ce_eu_pipeline message_bytes=",
                         r.message_bytes,
                         " chunk_bytes=",
                         plan.chunk_bytes,
                         " chunk_count=",
                         plan.num_chunks,
                         " slots=",
                         plan.slots,
                         " ce_queue_index=",
                         plan.ce_queue_ptr
                             ? std::to_string(plan.ce_ordinal) + "." + std::to_string(plan.ce_index)
                             : std::string("none"),
                         " ce_queue_note=",
                         plan.ce_note ? plan.ce_note : "none",
                         " compute_queue_index=",
                         stream_queue_index(r.stream),
                         " fallback=false");
            }
            else {
                LOG_INFO("collective=",
                         r.collective,
                         " ranks=",
                         r.ranks,
                         " selected_path=serial message_bytes=",
                         r.message_bytes,
                         " chunk_bytes=",
                         plan.chunk_bytes,
                         " chunk_count=",
                         plan.num_chunks,
                         " ce_queue_index=none compute_queue_index=",
                         stream_queue_index(r.stream),
                         " fallback=true fallback_reason=",
                         reason);
            }
        }
    }
    return plan;
}
