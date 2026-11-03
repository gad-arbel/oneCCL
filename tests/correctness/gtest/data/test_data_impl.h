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

#include "bf16.h"
#include "compare/mismatch_report.h"
#include "data_spec.h"
#include "fold.h"
#include "host_buffer.h"
#include "orders/reduction_order.h"
#include "patterns/data_pattern.h"
#include "source_map.h"
#include "test_data.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace oneccl_test {

// Template implementation of ITestData for dtype T.
// Header-only, included only by test_data.cpp (the single dtype switch).
template <typename T> class TestDataImpl : public ITestData {
    // my_position_ value when my_rank is not in comm_ranks
    static constexpr size_t kNoPosition = static_cast<size_t>(-1);

  public:
    explicit TestDataImpl(const DataSpec &spec,
                          std::unique_ptr<IDataPattern> pattern,
                          std::unique_ptr<IReductionOrder> order,
                          CompareMode mode)
        : spec_(spec), pattern_(std::move(pattern)), order_(std::move(order)),
          mode_(mode), rank_count_(spec.comm_ranks.size()),
          is_pre_mul_sum_(spec.op == PatternOp::ONECCL_PRE_MUL_SUM),
          needs_absolute_sum_(mode == CompareMode::ONECCL_TOLERANCE &&
                              is_absolute_sum_op(spec.op)) {

        // Allocate scratch once at construction
        pattern_values_.resize(kWindowElementCount);
        expected_values_.resize(kWindowElementCount);
        if (mode_ == CompareMode::ONECCL_TOLERANCE) {
            absolute_sums_.resize(kWindowElementCount);
        }
        accumulators_.resize(kWindowElementCount);
        permutation_.resize(rank_count_);

        // Resolve my_rank → position once. Invariant 1 is refused here, with
        // the same contract exception check_invariants() throws, rather than
        // through a gtest macro: the object would be unusable either way.
        my_position_ = kNoPosition;
        for (size_t position = 0; position < rank_count_; ++position) {
            if (spec_.comm_ranks[position] == spec_.my_rank) {
                my_position_ = position;
                break;
            }
        }
        if (my_position_ == kNoPosition) {
            throw std::logic_error("my_rank=" + std::to_string(spec_.my_rank) +
                                   " not in comm_ranks");
        }
    }

    void fill_send(HostBuffer &buffer) const override {
        const size_t element_count = buffer.size() / sizeof(T);
        T *values = buffer.as<T>();

        // Pattern values for my_rank at indices [0, element_count)
        std::vector<double> window_values(
            std::min(element_count, kWindowElementCount));

        // Iterate over the buffer in windows to bound the scratch size
        for (size_t window_offset = 0; window_offset < element_count;
             window_offset += window_values.size()) {
            size_t window_count =
                std::min(window_values.size(), element_count - window_offset);
            pattern_->fill_values(spec_.my_rank, window_offset, window_count,
                                  window_values.data());

            // Iterate over the values within the window and cast to T
            for (size_t index = 0; index < window_count; ++index) {
                values[window_offset + index] = cast(window_values[index]);
            }
        }
    }

    void poison_recv(HostBuffer &buffer) const override {
        std::memset(buffer.data(), static_cast<int>(kPoisonByte),
                    buffer.size());
    }

    void assert_still_poisoned(const HostBuffer &buffer) const override {
        const std::byte *bytes = buffer.data();
        const size_t byte_count = buffer.size();

        for (size_t byte_index = 0; byte_index < byte_count; ++byte_index) {
            if (bytes[byte_index] != kPoisonByte) {
                ADD_FAILURE() << "Recv buffer not poisoned at byte "
                              << byte_index << " (expected 0x" << std::hex
                              << static_cast<int>(kPoisonByte) << ", got 0x"
                              << static_cast<int>(bytes[byte_index]) << ")";
                return;
            }
        }
    }

    size_t compare(const HostBuffer &buffer, const SourceMap &source_map,
                   size_t offset, MismatchReport &report) const override {
        // Invariant 8: buffers cover [0,R) in order, with no gap and no
        // overlap. offset == 0 always starts a fresh verification pass (mirrors
        // check_invariants() below, which likewise only fires at offset == 0),
        // so it never conflicts with a previous pass's tracked offset.
        if (offset != 0 && offset != expected_next_offset_) {
            throw std::logic_error(
                "compare(): offset=" + std::to_string(offset) +
                " does not match expected_next_offset_=" +
                std::to_string(expected_next_offset_));
        }

        if (buffer.size() % sizeof(T) != 0) {
            throw std::logic_error(
                "compare(): buffer size=" + std::to_string(buffer.size()) +
                " bytes is not a multiple of element size=" +
                std::to_string(sizeof(T)));
        }

        const T *actual = buffer.as<T>();
        const size_t buffer_element_count = buffer.size() / sizeof(T);

        // Buffers must not run past the declared receive range (written so
        // that neither side can wrap)
        if (offset > spec_.recv_element_count ||
            buffer_element_count > spec_.recv_element_count - offset) {
            throw std::logic_error(
                "compare(): offset=" + std::to_string(offset) +
                " + buffer_element_count=" +
                std::to_string(buffer_element_count) +
                " exceeds recv_element_count=" +
                std::to_string(spec_.recv_element_count));
        }

        // Check invariants once on the first buffer (offset == 0)
        if (offset == 0) {
            check_invariants(source_map);
        }

        size_t mismatches = 0;
        const size_t buffer_limit = offset + buffer_element_count;

        // Window loop: process up to kWindowElementCount at a time
        for (size_t window_offset = offset; window_offset < buffer_limit;) {
            size_t window_end =
                std::min(window_offset + kWindowElementCount, buffer_limit);

            // Clip to segment boundary for RANK_PER_SEGMENT
            if (source_map.contributors ==
                Contributors::ONECCL_RANK_PER_SEGMENT) {
                window_end = std::min(window_end,
                                      segment_end(source_map, window_offset));
            }

            const size_t window_count = window_end - window_offset;

            // Build expected for [window_offset, window_end)
            fill_expected(source_map, window_offset, window_count);

            // Fast path: memcmp
            const T *actual_window = actual + (window_offset - offset);
            if (std::memcmp(actual_window, expected_values_.data(),
                            window_count * sizeof(T)) == 0) {
                window_offset = window_end;
                continue;
            }

            // Mismatch found - check mode
            if (mode_ == CompareMode::ONECCL_BIT_EXACT) {
                mismatches +=
                    compare_exact(actual_window, expected_values_.data(),
                                  window_count, window_offset, report);
            } else {
                // Only a fold has an absolute sum. A single contribution is
                // one cast, so its window is u * |expected| alone.
                const double *absolute_sums =
                    (needs_absolute_sum_ &&
                     source_map.contributors == Contributors::ONECCL_ALL_RANKS)
                        ? absolute_sums_.data()
                        : nullptr;
                mismatches += compare_tolerance(
                    actual_window, expected_values_.data(), absolute_sums,
                    window_count, window_offset, report);
            }

            window_offset = window_end;
        }

        expected_next_offset_ = offset + buffer_element_count;
        return mismatches;
    }

    void verify(const HostBuffer &buffer, const SourceMap &source_map,
                size_t offset) const override {
        MismatchReport report;
        const size_t mismatch_count =
            compare(buffer, source_map, offset, report);
        if (mismatch_count > 0) {
            report.fail_now_with_report(spec_.my_rank);
        }
    }

  private:
    // Ops whose tolerance bound needs the accumulated absolute-value sum
    static bool is_absolute_sum_op(PatternOp op) {
        return op == PatternOp::ONECCL_SUM ||
               op == PatternOp::ONECCL_PRE_MUL_SUM ||
               op == PatternOp::ONECCL_AVG;
    }

    // The single double→T cast site
    T cast(double value) const {
        return numeric_conversion::from_double<T>(value);
    }

    // Build expected values for the window [offset, offset+count) into
    // expected_values_, and for a fold the absolute sums into absolute_sums_
    void fill_expected(const SourceMap &source_map, size_t offset,
                       size_t count) const {
        T *out_values = expected_values_.data();
        switch (source_map.contributors) {
        case Contributors::ONECCL_SINGLE_RANK: {
            // One fixed rank
            pattern_->fill_values(source_map.root,
                                  source_map.source_base + offset, count,
                                  pattern_values_.data());
            for (size_t index = 0; index < count; ++index) {
                out_values[index] = cast(pattern_values_[index]);
            }
            break;
        }

        case Contributors::ONECCL_RANK_PER_SEGMENT: {
            // One rank per segment
            auto [segment, offset_in_segment] = locate(source_map, offset);
            const int source_rank = spec_.comm_ranks[segment];
            const size_t source_offset = source_offset_for(source_map, segment);
            pattern_->fill_values(source_rank,
                                  source_offset + offset_in_segment, count,
                                  pattern_values_.data());
            for (size_t index = 0; index < count; ++index) {
                out_values[index] = cast(pattern_values_[index]);
            }
            break;
        }

        case Contributors::ONECCL_ALL_RANKS: {
            // Fold over all ranks in order
            order_->order(spec_.my_rank, rank_count_, permutation_);

            // Initialize absolute sums if needed
            if (needs_absolute_sum_) {
                std::fill_n(absolute_sums_.begin(), count, 0.0);
            }

            bool is_first_contributor = true;
            for (size_t position : permutation_) {
                const int rank = spec_.comm_ranks[position];
                // The product multiplies a T send value by a T scalar, so
                // both operands are cast to T before the multiply
                const double scalar =
                    is_pre_mul_sum_ ? numeric_conversion::to_double(
                                          cast(spec_.pre_mul_scalars[position]))
                                    : 1.0;

                pattern_->fill_values(rank, source_map.source_base + offset,
                                      count, pattern_values_.data());

                // Accumulate contributions into T and absolute sums
                for (size_t index = 0; index < count; ++index) {
                    const T contribution =
                        cast(scalar * numeric_conversion::to_double(
                                          cast(pattern_values_[index])));

                    if (is_first_contributor) {
                        accumulators_[index] = Fold<T>::init(contribution);
                    } else {
                        Fold<T>::step(spec_.op, accumulators_[index],
                                      contribution);
                    }

                    if (needs_absolute_sum_) {
                        Fold<T>::accumulate_absolute(absolute_sums_[index],
                                                     contribution);
                    }
                }

                is_first_contributor = false;
            }

            // Finalize to T
            for (size_t index = 0; index < count; ++index) {
                out_values[index] = Fold<T>::finish(
                    spec_.op, accumulators_[index], rank_count_);
            }
            break;
        }
        }
    }

    // Compare exact (bit-exact mode)
    size_t compare_exact(const T *actual, const T *expected, size_t count,
                         size_t base_index, MismatchReport &report) const {
        size_t mismatches = 0;
        for (size_t offset = 0; offset < count; ++offset) {
            if (actual[offset] != expected[offset]) {
                report.add_single_mismatch(
                    base_index + offset,
                    numeric_conversion::to_double(actual[offset]),
                    numeric_conversion::to_double(expected[offset]), 0.0);
                ++mismatches;
            }
        }
        return mismatches;
    }

    // Compare with tolerance
    size_t compare_tolerance(const T *actual, const T *expected,
                             const double *absolute_sums, size_t count,
                             size_t base_index, MismatchReport &report) const {
        size_t mismatches = 0;
        for (size_t offset = 0; offset < count; ++offset) {
            const double tolerance = ITestData::tolerance_bound(
                spec_.dtype, rank_count_,
                absolute_sums ? absolute_sums[offset] : 0.0,
                numeric_conversion::to_double(expected[offset]));
            const double difference =
                std::fabs(numeric_conversion::to_double(actual[offset]) -
                          numeric_conversion::to_double(expected[offset]));

            // NOT `> tolerance`: treats NaN as mismatch
            if (!(difference <= tolerance)) {
                report.add_single_mismatch(
                    base_index + offset,
                    numeric_conversion::to_double(actual[offset]),
                    numeric_conversion::to_double(expected[offset]), tolerance);
                ++mismatches;
            }
        }
        return mismatches;
    }

    // Check invariants for source_map and spec. A plain contract check (not a
    // gtest macro): compare() must stay usable by non-gtest callers, so a
    // violation throws std::logic_error on the first failing invariant
    // instead of gtest's collect-all-then-continue style.
    void check_invariants(const SourceMap &source_map) const {
        const size_t recv_element_count = spec_.recv_element_count;
        const size_t send_element_count = spec_.send_element_count;

        // Invariant 1: my_rank is in comm_ranks (also refused at construction)
        if (my_position_ == kNoPosition) {
            throw std::logic_error("my_rank=" + std::to_string(spec_.my_rank) +
                                   " not in comm_ranks");
        }

        switch (source_map.contributors) {
        case Contributors::ONECCL_ALL_RANKS:
            // Invariant 2
            if (recv_element_count != spec_.count) {
                throw std::logic_error(
                    "ALL_RANKS: recv_element_count=" +
                    std::to_string(recv_element_count) +
                    " must equal count=" + std::to_string(spec_.count));
            }
            if (source_map.segment_element_count != 0u) {
                throw std::logic_error(
                    "ALL_RANKS: segment_element_count=" +
                    std::to_string(source_map.segment_element_count) +
                    " must be 0");
            }
            if (!source_map.segment_offsets.empty()) {
                throw std::logic_error(
                    "ALL_RANKS: segment_offsets must be empty");
            }
            if (!source_map.source_offsets.empty()) {
                throw std::logic_error(
                    "ALL_RANKS: source_offsets must be empty");
            }
            // Invariant 6: max send index < send_element_count
            if (!fits_in_send(source_map.source_base, recv_element_count)) {
                throw std::logic_error(
                    "ALL_RANKS: source index out of bounds: source_base=" +
                    std::to_string(source_map.source_base) +
                    " recv_element_count=" +
                    std::to_string(recv_element_count) +
                    " send_element_count=" +
                    std::to_string(send_element_count));
            }
            break;

        case Contributors::ONECCL_RANK_PER_SEGMENT:
            if (source_map.segment_element_count > 0) {
                // Invariant 3: uniform segments
                if (source_map.segment_element_count >
                        recv_element_count / std::max<size_t>(rank_count_, 1) ||
                    recv_element_count !=
                        rank_count_ * source_map.segment_element_count) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT uniform: recv_element_count=" +
                        std::to_string(recv_element_count) +
                        " != rank_count*segment_element_count=" +
                        std::to_string(rank_count_ *
                                       source_map.segment_element_count));
                }
                if (!source_map.segment_offsets.empty()) {
                    throw std::logic_error("RANK_PER_SEGMENT uniform: "
                                           "segment_offsets must be empty");
                }
                if (!source_map.source_offsets.empty()) {
                    throw std::logic_error("RANK_PER_SEGMENT uniform: "
                                           "source_offsets must be empty");
                }
                // Invariant 6
                if (!fits_in_send(source_map.source_base,
                                  source_map.segment_element_count)) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT uniform: source index out of bounds: "
                        "source_base=" +
                        std::to_string(source_map.source_base) +
                        " segment_element_count=" +
                        std::to_string(source_map.segment_element_count) +
                        " send_element_count=" +
                        std::to_string(send_element_count));
                }
            } else {
                // Invariant 4: variable segments
                if (source_map.source_base != 0u) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT variable: source_base=" +
                        std::to_string(source_map.source_base) +
                        " must be 0 (source_offsets carries the bases)");
                }
                if (source_map.segment_offsets.size() != rank_count_ + 1) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT variable: segment_offsets size=" +
                        std::to_string(source_map.segment_offsets.size()) +
                        " must be rank_count+1=" +
                        std::to_string(rank_count_ + 1));
                }
                if (source_map.source_offsets.size() != rank_count_) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT variable: source_offsets size=" +
                        std::to_string(source_map.source_offsets.size()) +
                        " must be rank_count=" + std::to_string(rank_count_));
                }
                if (source_map.segment_offsets[0] != 0u) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT variable: segment_offsets[0]=" +
                        std::to_string(source_map.segment_offsets[0]) +
                        " must be 0");
                }
                if (source_map.segment_offsets[rank_count_] !=
                    recv_element_count) {
                    throw std::logic_error(
                        "RANK_PER_SEGMENT variable: "
                        "segment_offsets[rank_count]=" +
                        std::to_string(
                            source_map.segment_offsets[rank_count_]) +
                        " must be recv_element_count=" +
                        std::to_string(recv_element_count));
                }
                // Check non-decreasing
                for (size_t segment = 1; segment <= rank_count_; ++segment) {
                    if (source_map.segment_offsets[segment - 1] >
                        source_map.segment_offsets[segment]) {
                        throw std::logic_error(
                            "RANK_PER_SEGMENT variable: segment_offsets not "
                            "non-decreasing at " +
                            std::to_string(segment));
                    }
                }
                // Invariant 6: each source segment lies inside the send buffer
                for (size_t segment = 0; segment < rank_count_; ++segment) {
                    const size_t segment_length =
                        source_map.segment_offsets[segment + 1] -
                        source_map.segment_offsets[segment];
                    if (!fits_in_send(source_map.source_offsets[segment],
                                      segment_length)) {
                        throw std::logic_error(
                            "RANK_PER_SEGMENT variable: source index out of "
                            "bounds at segment " +
                            std::to_string(segment) + ": source_offset=" +
                            std::to_string(source_map.source_offsets[segment]) +
                            " segment_length=" +
                            std::to_string(segment_length) +
                            " send_element_count=" +
                            std::to_string(send_element_count));
                    }
                }
            }
            break;

        case Contributors::ONECCL_SINGLE_RANK:
            // Invariant 5
            if (source_map.root < 0) {
                throw std::logic_error(
                    "SINGLE_RANK: root=" + std::to_string(source_map.root) +
                    " must be >= 0");
            }
            if (std::find(spec_.comm_ranks.begin(), spec_.comm_ranks.end(),
                          source_map.root) == spec_.comm_ranks.end()) {
                throw std::logic_error(
                    "SINGLE_RANK: root=" + std::to_string(source_map.root) +
                    " not in comm_ranks");
            }
            if (recv_element_count != spec_.count) {
                throw std::logic_error(
                    "SINGLE_RANK: recv_element_count=" +
                    std::to_string(recv_element_count) +
                    " must equal count=" + std::to_string(spec_.count));
            }
            if (source_map.segment_element_count != 0u ||
                !source_map.segment_offsets.empty() ||
                !source_map.source_offsets.empty()) {
                throw std::logic_error(
                    "SINGLE_RANK: segment_element_count, segment_offsets and "
                    "source_offsets must be unset");
            }
            // Invariant 6
            if (!fits_in_send(source_map.source_base, recv_element_count)) {
                throw std::logic_error(
                    "SINGLE_RANK: source index out of bounds: source_base=" +
                    std::to_string(source_map.source_base) +
                    " recv_element_count=" +
                    std::to_string(recv_element_count) +
                    " send_element_count=" +
                    std::to_string(send_element_count));
            }
            break;
        }

        // Invariant 7: pre_mul_scalars size
        if (spec_.op == PatternOp::ONECCL_PRE_MUL_SUM) {
            if (spec_.pre_mul_scalars.size() != rank_count_) {
                throw std::logic_error(
                    "PRE_MUL_SUM: pre_mul_scalars size=" +
                    std::to_string(spec_.pre_mul_scalars.size()) +
                    " must equal rank_count=" + std::to_string(rank_count_));
            }
        } else {
            if (!spec_.pre_mul_scalars.empty()) {
                throw std::logic_error(
                    "Non-PRE_MUL_SUM: pre_mul_scalars must be empty");
            }
        }
    }

    // True when [begin, begin + length) lies inside the send buffer, written
    // so that neither the sum nor the bound can wrap
    bool fits_in_send(size_t begin, size_t length) const {
        return begin <= spec_.send_element_count &&
               length <= spec_.send_element_count - begin;
    }

    // Return the end of the segment that holds index
    size_t segment_end(const SourceMap &source_map, size_t index) const {
        if (source_map.segment_element_count > 0) {
            // Uniform: next segment boundary
            return ((index / source_map.segment_element_count) + 1) *
                   source_map.segment_element_count;
        } else {
            // Variable: next offset
            const size_t segment = segment_index(source_map, index);
            return source_map.segment_offsets[segment + 1];
        }
    }

    // Locate segment and offset within segment for index
    std::pair<size_t, size_t> locate(const SourceMap &source_map,
                                     size_t index) const {
        if (source_map.segment_element_count > 0) {
            // Uniform: (segment index, offset in segment)
            return {index / source_map.segment_element_count,
                    index % source_map.segment_element_count};
        } else {
            // Variable: binary search
            const size_t segment = segment_index(source_map, index);
            return {segment, index - source_map.segment_offsets[segment]};
        }
    }

    // Return segment index for index (variable segments only)
    size_t segment_index(const SourceMap &source_map, size_t index) const {
        // Binary search for segment with
        // segment_offsets[segment] <= index < segment_offsets[segment+1]
        auto boundary =
            std::upper_bound(source_map.segment_offsets.begin(),
                             source_map.segment_offsets.end(), index);
        return static_cast<size_t>(boundary -
                                   source_map.segment_offsets.begin()) -
               1;
    }

    // Return the send-buffer offset segment reads from
    size_t source_offset_for(const SourceMap &source_map,
                             size_t segment) const {
        if (source_map.segment_element_count > 0) {
            // Uniform: constant source_base
            return source_map.source_base;
        } else {
            // Variable: per-rank offset
            return source_map.source_offsets[segment];
        }
    }

    // Members
    DataSpec spec_;
    std::unique_ptr<IDataPattern> pattern_;
    std::unique_ptr<IReductionOrder> order_;
    CompareMode mode_;
    size_t rank_count_;       // comm_ranks.size()
    size_t my_position_;      // position of my_rank in comm_ranks
    bool is_pre_mul_sum_;     // op == PRE_MUL_SUM
    bool needs_absolute_sum_; // tolerance + (sum|pre_mul_sum|avg)

    // Scratch arrays, sized once at construction. They and
    // expected_next_offset_ make compare() and verify() unsafe to call
    // concurrently on one instance: each rank or thread owns its own.
    mutable std::vector<double> pattern_values_; // pattern values
    mutable std::vector<T> expected_values_;     // expected values in T
    mutable std::vector<double> absolute_sums_;  // sum of |contribution|
    mutable std::vector<typename Fold<T>::Accumulator>
        accumulators_;                        // accumulators
    mutable std::vector<size_t> permutation_; // rank permutation

    // Unlike the vectors above (genuine caches), this is a correctness
    // contract, not thread-safe to mutate concurrently.
    mutable size_t expected_next_offset_ =
        0; // next expected compare() offset (invariant 8)
};

} // namespace oneccl_test
