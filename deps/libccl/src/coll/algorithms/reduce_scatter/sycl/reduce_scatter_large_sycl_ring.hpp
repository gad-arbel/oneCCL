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
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/selection/policy/policy_env.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_ce_eu_pipeline.hpp"
//#include "coll/algorithms/utils/sycl_ll256.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

#if 0
template <typename T>
inline void reduce_kernel(void *out, void *in1, void *in2, size_t idx) {
    ((T *)out)[idx] = ((T *)in1)[idx] + ((T *)in2)[idx];
}
#endif

// Reduce one element at an explicit index, so a caller spreading its work over several
// independent work-groups can index by the group-local id. Same split as copy_at().
template <typename T, int vec_size>
void inline reduce_at(void *out,
                      void *in1,
                      void *in2,
                      const size_t count,
                      const size_t idx,
                      ccl_reduction_data reduction) {
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        //reduce_kernel<AT>(out, in1, in2, idx);
        reduce_pair_dispatch<AT>(in1, in2, out, reduction, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            //reduce_kernel<T>(out, in1, in2, new_idx);
            reduce_pair_dispatch<T>(in1, in2, out, reduction, new_idx);
        }
    }
}

template <typename T, int vec_size>
void inline reduce(void *out,
                   void *in1,
                   void *in2,
                   const size_t count,
                   const size_t offset,
                   ccl_reduction_data reduction,
                   const sycl::nd_item<1> it) {
    reduce_at<T, vec_size>(out, in1, in2, count, it.get_global_linear_id() + offset, reduction);
}

template <typename T>
inline void reduce_copy_kernel(void *out1,
                               void *in1,
                               void *in2,
                               void *out2,
                               ccl_reduction_data reduction,
                               size_t idx) {
    //reduce_kernel<T>(out1, in1, in2, idx);
    reduce_pair_dispatch<T>(in1, in2, out1, reduction, idx);
    copy_kernel<T>(out2, out1, idx);
}

// Explicit-index form, see reduce_at().
template <typename T, int vec_size>
void inline reduce_copy_at(void *out1,
                           void *in1,
                           void *in2,
                           void *out2,
                           const size_t count,
                           const size_t idx,
                           ccl_reduction_data reduction) {
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        reduce_copy_kernel<AT>(out1, in1, in2, out2, reduction, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_copy_kernel<T>(out1, in1, in2, out2, reduction, new_idx);
        }
    }
}

template <typename T, int vec_size>
void inline reduce_copy(void *out1,
                        void *in1,
                        void *in2,
                        void *out2,
                        const size_t count,
                        const size_t offset,
                        ccl_reduction_data reduction,
                        const sycl::nd_item<1> it) {
    reduce_copy_at<T, vec_size>(out1, in1, in2, out2, count, it.get_global_linear_id() + offset, reduction);
}

template <typename T, int full_vec_size, int partial_vec_size>
struct CopyKernel {
    CopyKernel(void *dst_ptr, const void *src_ptr, const size_t count, ccl_comm *node_comm, int use_full_vector)
            : dst_ptr(dst_ptr),
              src_ptr(src_ptr),
              count(count),
              flag_data(node_comm->flag_data()),
              use_full_vector(use_full_vector) {}

    void operator()(sycl::nd_item<1> it) const {
        copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst_ptr, src_ptr, count, use_full_vector);
        p2p_barrier(flag_data, it, false /* use_subgroups */, true /* use_root_sync */);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    void *dst_ptr;
    const void *src_ptr;
    size_t count;
    ccl_comm_flag_data flag_data;
    int use_full_vector;
};

template <typename T, int full_vec_size, int partial_vec_size>
#if !RS_USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline reduce_kernel_wrap(void *out, void *in1, void *in2, const size_t count, ccl_reduction_data reduction, int use_full_vector) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;
    const size_t req_threads = count / vec_size + count % vec_size;

    if (use_full_vector) {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            reduce<T, full_vec_size>(out, in1, in2, count, offset, reduction, it);
        }
    }
    else {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            reduce<T, partial_vec_size>(out, in1, in2, count, offset, reduction, it);
        }
    }
}

template <typename T, int vec_size>
struct ReduceKernel {
    ReduceKernel(void *out, void *in1, void *in2, const size_t count, ccl_reduction_data reduction)
            : out(out),
              in1(in1),
              in2(in2),
              count(count),
              reduction(reduction) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_kernel_wrap<T, vec_size>(out, in1, in2, count, reduction);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16> };
    }

    void *out;
    void *in1;
    void *in2;
    size_t count;
    ccl_reduction_data reduction;
};

template <typename T, int full_vec_size, int partial_vec_size>
#if !RS_USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline reduce_copy_kernel_wrap(void *out1,
                                    void *in1,
                                    void *in2,
                                    void *out2,
                                    const size_t count,
                                    ccl_reduction_data reduction,
                                    int use_full_vector) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;
    const size_t req_threads = count / vec_size + count % vec_size;

    if (use_full_vector) {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            reduce_copy<T, full_vec_size>(out1, in1, in2, out2, count, offset, reduction, it);
        }
    }
    else {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            reduce_copy<T, partial_vec_size>(out1, in1, in2, out2, count, offset, reduction, it);
        }
    }
}

template <typename T, int vec_size>
struct ReduceCopyKernel {
    ReduceCopyKernel(void *out1,
                     void *in1,
                     void *in2,
                     void *out2,
                     const size_t count,
                     ccl_reduction_data reduction)
            : out1(out1),
              in1(in1),
              in2(in2),
              out2(out2),
              count(count),
              reduction(reduction) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_copy_kernel_wrap<T, vec_size, vec_size>(out1, in1, in2, out2, count, reduction, true);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16> };
    }

    void *out1;
    void *in1;
    void *in2;
    void *out2;
    size_t count;
    ccl_reduction_data reduction;
};

// ---- CE/EU pipeline: copy-engine copies overlapped with EU reductions ----
// (here because it uses reduce_kernel_wrap; the allreduce ring includes this header)

// The arguments of a pipelined collective.
struct ce_eu_args_t {
    bool is_allreduce = false;
    const void *send_buf = nullptr;
    void *recv_buf = nullptr;
    size_t count_per_rank = 0;
    size_t send_stride = 0; // elements between the parts in send_buf
    ccl::datatype dtype{};
    ccl::reduction reduction{};
    ccl_comm *comm = nullptr;
    std::shared_ptr<ccl_comm> p2p_comm;
    ccl_stream *global_stream = nullptr;
    void *peer_recv_buf = nullptr; // the right neighbour's recv_buf (allreduce only)
    const ccl::vector_class<ccl::event> *deps_ptr = nullptr;
};

// This rank's slots (written by the left neighbour) and the right neighbour's.
struct ce_eu_slots_t {
    std::array<void *, CE_EU_MAX_SLOTS> local{};
    std::array<void *, CE_EU_MAX_SLOTS> remote{};
};

inline ce_eu_slots_t make_ce_eu_slots(const ce_eu_args_t &args,
                                      int remote_rank,
                                      const ce_eu_pipeline_plan_t &plan) {
    ce_eu_slots_t slots;
    slots.local[0] = get_large_tmp_buf(0, args.comm);
    slots.remote[0] = get_remote_large_tmp_buf(0, args.p2p_comm.get())[remote_rank];
    for (size_t i = 1; i < plan.slots; i++) {
        slots.local[i] = static_cast<char *>(slots.local[i - 1]) + plan.chunk_bytes;
        slots.remote[i] = static_cast<char *>(slots.remote[i - 1]) + plan.chunk_bytes;
    }
    return slots;
}

// out = in1 (+) in2 for one chunk, on the stream queue
template <typename T>
class ce_eu_reducer {
public:
    ce_eu_reducer(const sycl::queue &queue, ccl::reduction reduction, bool use_full_vector)
            : m_queue(queue),
              m_reduction_op(make_reduction_operation(reduction)),
              m_use_full_vector(use_full_vector),
              m_num_threads(ccl::global_data::env().sycl_num_threads),
              m_wg_size(ccl::global_data::env().sycl_work_group_size) {
        CCL_THROW_IF_NOT(
            m_wg_size > 0 && m_wg_size <= m_queue.get_device().get_info<sycl::info::device::max_work_group_size>(),
            "invalid CCL_SYCL_WORK_GROUP_SIZE: ",
            m_wg_size);
    }

    sycl::event operator()(void *out, const void *in1, void *in2, size_t count) {
        constexpr int full_vec_size = get_num_elements<T, CE_EU_VECTOR_BYTES, true>();
        constexpr int partial_vec_size = get_num_elements<T, CE_EU_VECTOR_BYTES, false>();
        const int vec_size = m_use_full_vector ? full_vec_size : partial_vec_size;

        size_t kernel_threads = count / vec_size + count % vec_size;
        if (m_num_threads) {
            kernel_threads = std::min(kernel_threads, m_num_threads);
        }
        const size_t kernel_size = ((kernel_threads + m_wg_size - 1) / m_wg_size) * m_wg_size;
        sycl::nd_range<1> nd_range_l(kernel_size, m_wg_size);

        void *in1_ptr = const_cast<void *>(in1);
        const bool use_full_vector = m_use_full_vector;
        const ccl_reduction_data op = m_reduction_op;
        return m_queue.parallel_for(nd_range_l,
                                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(CE_EU_SUB_GROUP_SIZE)]] {
                                        reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(
                                            out, in1_ptr, in2, count, op, use_full_vector);
                                    });
    }

private:
    sycl::queue m_queue;
    ccl_reduction_data m_reduction_op;
    bool m_use_full_vector;
    size_t m_num_threads;
    size_t m_wg_size;
};

// What both pipelines derive from their arguments.
template <typename T>
struct ce_eu_context_t {
    ce_eu_context_t(const ce_eu_args_t &args, const ce_eu_pipeline_plan_t &plan)
            : dsize(ccl::global_data::get().dtypes->get(args.dtype).size()),
              part_bytes(args.count_per_rank * dsize),
              send_stride_bytes(args.send_stride * dsize),
              q(args.global_stream->get_native_stream()),
              q_ce(plan.ce_queue(q)),
              ranks(args.p2p_comm->size()),
              rank(args.p2p_comm->rank()),
              remote_rank((rank + 1) % ranks),
              reducer(q,
                      args.reduction,
                      can_use_full_vector(args.send_buf, args.recv_buf, args.count_per_rank, dsize) &&
                          can_use_full_vector(args.send_buf, args.recv_buf, args.send_stride, dsize)),
              chunks(plan, args.count_per_rank, dsize),
              slots(make_ce_eu_slots(args, remote_rank, plan)),
              num_slots(plan.slots),
              num_chunks(plan.num_chunks),
              // where this rank's reduced part goes
              out_part_ptr(static_cast<char *>(args.recv_buf) + (args.is_allreduce ? rank * part_bytes : 0)) {}

    const size_t dsize;
    const size_t part_bytes; // what each rank reduces
    const size_t send_stride_bytes; // distance between the parts in send_buf
    sycl::queue q; // barriers and reductions
    sycl::queue q_ce; // copies
    const int ranks;
    const int rank;
    const int remote_rank; // the right neighbour
    ce_eu_reducer<T> reducer;
    const ce_eu_chunking_t chunks;
    const ce_eu_slots_t slots;
    const size_t num_slots;
    const size_t num_chunks;
    char *out_part_ptr;
};

// ---- 2 ranks ----
//   CE : push 0 | push 1 | push 2 | result 0 | push 3 | result 1 | ...
//   EU :        barrier 0 | reduce 0 | barrier 1 | reduce 1 | ...
// (reduce-scatter: no result pushes). Push k reuses the slot of chunk k - slots, so it waits
// for barrier k - slots + 1, which the peer enters only after reducing that chunk.
template <typename T>
sycl::event two_rank_ce_eu_pipeline(ce_eu_context_t<T> &ctx, const ce_eu_args_t &args) {
    CCL_ASSERT(ctx.ranks == CE_EU_MIN_RANKS && ctx.num_slots >= CE_EU_MIN_SLOTS &&
               ctx.num_chunks >= CE_EU_MIN_CHUNKS);
    const bool is_allreduce = args.is_allreduce;
    const size_t dsize = ctx.dsize;
    sycl::queue &q = ctx.q;
    sycl::queue &q_ce = ctx.q_ce;
    const size_t num_slots = ctx.num_slots;
    const size_t num_chunks = ctx.num_chunks;
    const std::shared_ptr<ccl_comm> &p2p_comm = args.p2p_comm;

    // this rank's reduced part at the peer (allreduce)
    char *peer_out_part_ptr =
        is_allreduce ? static_cast<char *>(args.peer_recv_buf) + ctx.rank * ctx.part_bytes : nullptr;

    std::vector<sycl::event> p_ev(num_chunks); // push
    std::vector<sycl::event> b_ev(num_chunks); // barrier
    std::vector<sycl::event> r_ev(num_chunks); // reduction
    std::vector<sycl::event> res_ev(is_allreduce ? num_chunks : 0); // result push

    // our copy of the peer's part, into the peer's slot
    auto submit_push = [&](size_t c, const std::vector<sycl::event> &wait_for) {
        const char *src = static_cast<const char *>(args.send_buf) + ctx.remote_rank * ctx.send_stride_bytes +
                          ctx.chunks.offset_bytes(c);
        p_ev[c] = q_ce.memcpy(ctx.slots.remote[c % num_slots], src, ctx.chunks.elems(c) * dsize, wait_for);
    };

    std::vector<sycl::event> dep_events = get_sycl_events(*args.deps_ptr);
    sycl::event init_e = invoke_barrier(p2p_comm, q, dep_events, false);

    const size_t prefetch = std::min(num_slots, num_chunks);
    submit_push(0, { init_e });
    for (size_t c = 1; c < prefetch; c++) {
        submit_push(c, {});
    }

    b_ev[0] = invoke_p2p_barrier(p2p_comm, q, { p_ev[0] }, false);
    for (size_t c = 0; c < num_chunks; c++) {
        const size_t data_count = ctx.chunks.elems(c);
        void *out1 = ctx.out_part_ptr + ctx.chunks.offset_bytes(c);
        const void *in1 = static_cast<const char *>(args.send_buf) + ctx.rank * ctx.send_stride_bytes +
                          ctx.chunks.offset_bytes(c);
        void *in2 = ctx.slots.local[c % num_slots];

        r_ev[c] = ctx.reducer(out1, in1, in2, data_count);

        if (c + 1 < num_chunks) {
            b_ev[c + 1] = invoke_p2p_barrier(p2p_comm, q, { p_ev[c + 1] }, false);
        }

        if (is_allreduce) {
            res_ev[c] =
                q_ce.memcpy(peer_out_part_ptr + ctx.chunks.offset_bytes(c), out1, data_count * dsize, { r_ev[c] });
        }

        // the peer is done with slot c once barrier c + 1 passed
        if (c + num_slots < num_chunks) {
            submit_push(c + num_slots, { b_ev[c + 1] });
        }
    }

    if (!is_allreduce) {
        return r_ev[num_chunks - 1]; // the last operation on q
    }
    // waits for our last result push and tells us the peer's landed
    return invoke_p2p_barrier(p2p_comm, q, { res_ev[num_chunks - 1] }, false);
}

// ---- 3 to 5 ranks ----
// Chunk c does ring step i in time slot c + i (a wavefront). Per time slot: the sends on CE,
// then one barrier and the reductions on EU. The allreduce then runs the N - 2 allgather steps.
// Slot reuse: is_ring_slot_count_safe().
template <typename T>
sycl::event wavefront_ce_eu_pipeline(ce_eu_context_t<T> &ctx, const ce_eu_args_t &args) {
    CCL_ASSERT(ctx.ranks >= CE_EU_MIN_RING_RANKS && is_ring_slot_count_safe(ctx.ranks, ctx.num_slots) &&
               ctx.num_chunks >= CE_EU_MIN_CHUNKS);
    const bool is_allreduce = args.is_allreduce;
    const size_t dsize = ctx.dsize;
    sycl::queue &q = ctx.q;
    sycl::queue &q_ce = ctx.q_ce;
    const int N = ctx.ranks;
    const int rank = ctx.rank;
    const size_t steps = N - 1;
    const size_t num_slots = ctx.num_slots;
    const size_t num_chunks = ctx.num_chunks;
    const std::shared_ptr<ccl_comm> &p2p_comm = args.p2p_comm;

    auto ring_index = [&](int k) {
        return static_cast<size_t>((k % N + N) % N);
    };
    // slot of step i of chunk c, the same on every rank
    auto slot_of = [&](size_t c, size_t i) {
        return (c * steps + i) % num_slots;
    };
    // part this rank reduces in step i: the left neighbour starts with its own left part
    auto part_of = [&](size_t i) {
        return ring_index(rank - 2 - static_cast<int>(i));
    };

    // the allreduce pushes the finished part once more
    const size_t num_time_slots = num_chunks + steps - (is_allreduce ? 0 : 1);
    std::vector<sycl::event> r_ev(num_chunks * steps); // per (chunk, step)
    std::vector<sycl::event> b_ev(num_time_slots); // per time slot

    std::vector<sycl::event> dep_events = get_sycl_events(*args.deps_ptr);
    sycl::event init_e = invoke_barrier(p2p_comm, q, dep_events, false);

    bool is_first_push = true;
    std::vector<sycl::event> slot_pushes;
    auto push = [&](void *dst, const void *src, size_t bytes, std::vector<sycl::event> wait_for) {
        if (is_first_push) {
            wait_for.push_back(init_e);
            is_first_push = false;
        }
        slot_pushes.push_back(q_ce.memcpy(dst, src, bytes, wait_for));
    };

    for (size_t t = 0; t < num_time_slots; t++) {
        slot_pushes.clear();

        // first push of chunk t: needs only the barrier CE_EU_FIRST_PUSH_LAG time slots back
        if (t < num_chunks) {
            std::vector<sycl::event> wait;
            if (t >= CE_EU_FIRST_PUSH_LAG) {
                wait.push_back(b_ev[t - CE_EU_FIRST_PUSH_LAG]);
            }
            push(ctx.slots.remote[slot_of(t, 0)],
                 static_cast<const char *>(args.send_buf) + ring_index(rank - 1) * ctx.send_stride_bytes +
                     ctx.chunks.offset_bytes(t),
                 ctx.chunks.elems(t) * dsize,
                 wait);
        }
        // partial sum of step i - 1 of chunk t - i
        for (size_t i = 1; i < steps; i++) {
            if (t >= i && t - i < num_chunks) {
                const size_t c = t - i;
                push(ctx.slots.remote[slot_of(c, i)],
                     ctx.slots.local[slot_of(c, i - 1)],
                     ctx.chunks.elems(c) * dsize,
                     { r_ev[c * steps + i - 1] });
            }
        }
        // allreduce: finished part of chunk t - steps to the right neighbour's output
        if (is_allreduce && t >= steps && t - steps < num_chunks) {
            const size_t c = t - steps;
            push(static_cast<char *>(args.peer_recv_buf) + rank * ctx.part_bytes + ctx.chunks.offset_bytes(c),
                 ctx.out_part_ptr + ctx.chunks.offset_bytes(c),
                 ctx.chunks.elems(c) * dsize,
                 { r_ev[c * steps + steps - 1] });
        }

        // the left neighbour's pushes of this time slot landed
        b_ev[t] = invoke_p2p_barrier(p2p_comm, q, slot_pushes, false);

        for (size_t i = 0; i < steps; i++) {
            if (t < i || t - i >= num_chunks) {
                continue;
            }
            const size_t c = t - i;
            void *in2 = ctx.slots.local[slot_of(c, i)];
            const void *in1 = static_cast<const char *>(args.send_buf) + part_of(i) * ctx.send_stride_bytes +
                              ctx.chunks.offset_bytes(c);
            void *out1 =
                (i == steps - 1) ? static_cast<void *>(ctx.out_part_ptr + ctx.chunks.offset_bytes(c)) : in2;
            r_ev[c * steps + i] = ctx.reducer(out1, in1, in2, ctx.chunks.elems(c));
        }
    }

    if (!is_allreduce) {
        return r_ev[num_chunks * steps - 1]; // the last operation on q
    }

    // allgather: one finished part per step
    sycl::event last = b_ev[num_time_slots - 1];
    for (int j = 1; j <= N - 2; j++) {
        const size_t s = ring_index(rank - j);
        sycl::event copy_e = q_ce.memcpy(static_cast<char *>(args.peer_recv_buf) + s * ctx.part_bytes,
                                         static_cast<const char *>(args.recv_buf) + s * ctx.part_bytes,
                                         ctx.part_bytes,
                                         { last });
        last = invoke_p2p_barrier(p2p_comm, q, { copy_e }, false);
    }
    return last;
}

template <typename T>
sycl::event run_ce_eu_pipeline(const ce_eu_args_t &args, const ce_eu_pipeline_plan_t &plan) {
    CCL_ASSERT(plan.use_pipeline && plan.elems_per_chunk > 0 && args.deps_ptr);
    ce_eu_context_t<T> ctx(args, plan);
    return ctx.ranks < CE_EU_MIN_RING_RANKS ? two_rank_ce_eu_pipeline<T>(ctx, args)
                                            : wavefront_ce_eu_pipeline<T>(ctx, args);
}

template <typename T>
sycl::event reduce_scatter_large_su_ring_multi_kernels(const void *send_buf,
                                                       void *recv_buf,
                                                       size_t recv_count,
                                                       size_t send_stride,
                                                       ccl::datatype dtype,
                                                       ccl::reduction reduction,
                                                       ccl_comm *comm,
                                                       std::shared_ptr<ccl_comm> p2p_comm,
                                                       ccl_stream *global_stream,
                                                       sycl_ptrs_type &sycl_ptrs,
                                                       const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const size_t recv_bytes = recv_count * dsize;
    const size_t send_stride_bytes = send_stride * dsize;
    sycl::queue q = global_stream->get_native_stream();

    const int N = p2p_comm->size();
    const int rank = p2p_comm->rank();
    const int dest_rank = (rank + 1) % N;

    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, recv_count, dsize) &&
                           can_use_full_vector(send_buf, recv_buf, send_stride, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

#if !RS_USE_PARALLEL_FOR
    static auto exe_bndl_cp =
        syclexp::get_kernel_bundle<copy_kernel_wrap<T, full_vec_size, partial_vec_size>, sycl::bundle_state::executable>(q.get_context());
    static sycl::kernel k_func_cp = exe_bndl_cp.template ext_oneapi_get_kernel<copy_kernel_wrap<T, full_vec_size, partial_vec_size>>();

    static auto exe_bndl_rs =
        syclexp::get_kernel_bundle<reduce_kernel_wrap<T, full_vec_size, partial_vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_rs = exe_bndl_rs.template ext_oneapi_get_kernel<reduce_kernel_wrap<T, full_vec_size, partial_vec_size>>();

    static auto exe_bndl_rc =
        syclexp::get_kernel_bundle<reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_rc =
        exe_bndl_rc.template ext_oneapi_get_kernel<reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>>();
#endif

    {
        ce_eu_pipeline_request_t request;
        request.collective = "reduce_scatter";
        request.stream = global_stream;
        request.comm = comm;
        request.ranks = N;
        request.is_cpu_barrier = is_cpu_barrier;
        request.has_whole_vectors =
            is_whole_vector_count(recv_count, dsize) && is_whole_vector_count(send_stride, dsize);
        request.dsize = dsize;
        request.message_bytes = recv_bytes * N;
        request.count_per_rank = recv_count;
        request.preferred_chunk_bytes = CE_EU_REDUCE_SCATTER_CHUNK_BYTES;
        const ce_eu_pipeline_plan_t plan = plan_ce_eu_pipeline(request);
        if (plan.use_pipeline) {
            ce_eu_args_t args;
            args.is_allreduce = false;
            args.send_buf = send_buf;
            args.recv_buf = recv_buf;
            args.count_per_rank = recv_count;
            args.send_stride = send_stride;
            args.dtype = dtype;
            args.reduction = reduction;
            args.comm = comm;
            args.p2p_comm = p2p_comm;
            args.global_stream = global_stream;
            args.deps_ptr = &deps;
            std::vector<sycl::event> pipeline_evs;
            pipeline_evs.push_back(run_ce_eu_pipeline<T>(args, plan));
            if (reduction == ccl::reduction::avg) {
                return sycl_average(q, recv_buf, recv_count, p2p_comm->size(), dtype, pipeline_evs);
            }
            return submit_wait_on_events(q, pipeline_evs);
        }
    }

    // chunk size for 2 ranks is set to 16MB to allow more chunking and
    // pipelining to overlap copying and reduction.
    // only enable pipelining when the number of chunks is no less than 4.
    const int pipeline_size = std::max(N, ccl_large_tmp_bufs::buf_count);
    size_t chunk_size = get_simple_ring_segment_size(
        CCL_SYCL_REDUCE_SCATTER_SIMPLE_RING_SEGMENT_SIZE,
        ccl::selection::call_env().sycl_reduce_scatter_simple_ring_segment_size,
        pipeline_size,
        large_tmp_buf_align);
    const size_t chunk_size_for_2 = std::min(16777216UL, chunk_size);
    // recv_count is the chunk when the caller chunks; decide by the full block (send_stride)
    const bool use_memcpy = (N == 2 && send_stride_bytes >= 4 * chunk_size_for_2) ||
                            ccl::selection::call_env().sycl_copy_engine;
    if (N == 2 && use_memcpy)
        chunk_size = chunk_size_for_2;
    const size_t chunk_count = chunk_size / dsize;

    LOG_DEBUG("simple ring multiple kernel: rank: ", rank, " comm size: ", N, " recv_count: ", recv_count);

    const size_t rem_chunk_count = recv_count % chunk_count;
    const size_t num_chunks = recv_count / chunk_count + (rem_chunk_count != 0);

    std::array<void *, ARC_MAX_NUM> work_bufs;
    std::array<void *, ARC_MAX_NUM> remote_work_bufs;
    work_bufs[0] = get_large_tmp_buf(0, comm);
    remote_work_bufs[0] = get_remote_large_tmp_buf(0, p2p_comm.get())[dest_rank];
    for (int i = 1; i < pipeline_size; i++) {
        work_bufs[i] = (char *)work_bufs[i - 1] + chunk_size;
        remote_work_bufs[i] = (char *)remote_work_bufs[i - 1] + chunk_size;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t wg_size = ccl::global_data::env().sycl_work_group_size;
    assert(wg_size > 0 && wg_size <= max_work_group_size);
    const size_t sg_size = 16;

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    // create a different queue when copy engine is used
    sycl::queue q_red = use_memcpy ? sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{}) : q;

    sycl::event copy_e, barrier_e, red_e;

    // with use_memcpy the reductions run on q_red, which q does not wait for. Right after
    // each barrier the peer copies into our work_bufs[next_slot], so the barrier must also
    // wait for the local reduction that last read that slot; otherwise q runs ahead of q_red
    // and the peer overwrites data that has not been reduced yet (MLSL-4933)
    std::vector<sycl::event> slot_read_e(pipeline_size);
    std::vector<bool> slot_read_pending(pipeline_size, false);

    int slot = 0;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_count * dsize;
        const size_t data_count = (nc < recv_count / chunk_count) ? chunk_count : rem_chunk_count;

        // starting indexes
        int s = (rank - 1 + N) % N;
        int r = (s - 1 + N) % N;

        for (int i = 0; i < N - 1; i++) {
            int next_slot = (slot + 1) % pipeline_size;
            void *out1 = (i == N - 2) ? (char *)recv_buf + chunk_offset : work_bufs[slot];
            void *in1 = (char *)send_buf + r * send_stride_bytes + chunk_offset;
            void *in2 = work_bufs[slot];
            void *out2 = remote_work_bufs[next_slot];

            size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            if (num_threads) {
                kernel_threads = std::min(kernel_threads, num_threads);
            }
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
            sycl::nd_range<1> nd_range(kernel_size, wg_size);

            // copy to remote
            if (i == 0) {
                void *src = (char *)send_buf + s * send_stride_bytes + chunk_offset;
                void *dst = remote_work_bufs[slot];
                if (use_memcpy) {
                    copy_e = q.submit([=](sycl::handler& h) {
                       h.memcpy(dst, src, data_count * sizeof(T));
                    });
                }
                else {
#if RS_USE_PARALLEL_FOR
                    q.parallel_for(
                        nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                            copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst, src, data_count, use_full_vector);
                    });
#else
                    syclexp::nd_launch(q, nd_range, k_func_cp, dst, src, data_count);
#endif
                }
            }

            // barrier
            std::vector<sycl::event> barrier_deps;
            if (slot_read_pending[next_slot]) {
                barrier_deps.push_back(slot_read_e[next_slot]);
            }
            barrier_e = invoke_p2p_barrier(p2p_comm, q, barrier_deps, is_cpu_barrier);

            // local reduce
            if (i < N - 2) {
                sycl::nd_range<1> nd_range(kernel_size, wg_size);
                if (use_memcpy) {
#if RS_USE_PARALLEL_FOR
                    red_e = q_red.submit([=](sycl::handler& h) {
                        h.depends_on(barrier_e);
                        h.parallel_for(nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                            reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(out1, in1, in2, data_count, reduction_op, use_full_vector);
                        });
                    });
#else
                    syclexp::nd_launch(q, nd_range, k_func_rs, out1, in1, in2, data_count);
#endif
                    slot_read_e[slot] = red_e;
                    slot_read_pending[slot] = true;
                    copy_e = q.submit([=](sycl::handler& h) {
                        h.depends_on(red_e);
                        h.memcpy(out2, out1, data_count * sizeof(T));
                    });
                }
                else {
#if RS_USE_PARALLEL_FOR
                    q.parallel_for(
                        nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                            reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>(out1, in1, in2, out2, data_count, reduction_op, use_full_vector);
                    });
#else
                    syclexp::nd_launch(q, nd_range, k_func_rc, out1, in1, in2, out2, data_count, reduction_op);
#endif
                }
            }
            else {
#if RS_USE_PARALLEL_FOR
                red_e = q_red.submit([=](sycl::handler& h) {
                    if (use_memcpy) h.depends_on(barrier_e);
                    h.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
                        reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(out1, in1, in2, data_count, reduction_op, use_full_vector);
                    });
		});
                if (use_memcpy) {
                    slot_read_e[slot] = red_e;
                    slot_read_pending[slot] = true;
                }
#else
                syclexp::nd_launch(q, nd_range, k_func_rs, out1, in1, in2, data_count, reduction_op);
#endif
            }

            s = r;
            r = (s - 1 + N) % N;
            slot = next_slot;
        }
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(red_e);
        work_event = sycl_average(q, recv_buf, recv_count, p2p_comm->size(), dtype, evs);
    }
    else {
        std::vector<sycl::event> evs;
        if (use_memcpy) evs.push_back(red_e);
        work_event = submit_wait_on_events(q, evs);
    }

    return work_event;
}

constexpr int pipeline_size = ccl_large_tmp_bufs::buf_count;

struct reduce_scatter_coll_params {
    ccl_comm_flag_data flag = { -1, -1 };
    ccl_comm_barrier_data barrier_data = { -1, -1 };
    size_t chunk_size;
    std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
};

template <typename T, int vec_size>
#if !RS_USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline reduce_scatter_kernel_wrap(const void *send_buf,
                                       void *recv_buf,
                                       size_t recv_count,
                                       size_t send_stride,
                                       ccl_reduction_data reduction,
                                       const reduce_scatter_coll_params params) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);

    const int comm_rank = params.flag.rank();
    const int comm_size = params.flag.size();

    const int dsize = sizeof(T);
    const size_t recv_bytes = recv_count * dsize;
    const size_t send_stride_bytes = send_stride * dsize;
    const size_t chunk_size = params.chunk_size;
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);
    std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
    work_bufs = params.work_bufs;
    remote_work_bufs = params.remote_work_bufs;

    // use_root_sync is true
    comm_barrier<true>(params.barrier_data, it, true, true /* gpu_counter_increase */);

    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_size;
        const size_t count = ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t req_threads = count / vec_size + count % vec_size;

        // starting indexes
        int s = (comm_rank - 1 + comm_size) % comm_size;
        int r = (s - 1 + comm_size) % comm_size;

        for (int i = 0; i < comm_size - 1; i++) {
            void *src =
                (i == 0) ? (char *)send_buf + s * send_stride_bytes + chunk_offset : work_bufs[(i - 1) % pipeline_size];
            void *dst = remote_work_bufs[i % pipeline_size];
            void *out1 = (i == comm_size - 2) ? (char *)recv_buf + chunk_offset : work_bufs[i % pipeline_size];
            void *in1 = (char *)send_buf + r * send_stride_bytes + chunk_offset;
            void *in2 = work_bufs[i % pipeline_size];
            void *out2 = remote_work_bufs[(i + 1) % pipeline_size];

            if (i == 0) {
                void *src = (char *)send_buf + s * send_stride_bytes + chunk_offset;
                void *dst = remote_work_bufs[i % pipeline_size];
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    copy<T, vec_size>(dst, src, count, offset, it);
                }
            }

            p2p_barrier(params.flag, it, false, true);

            if (i < comm_size - 2) {
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce<T, vec_size>(out1, in1, in2, count, offset, reduction, it);
                    copy<T, vec_size>(out2, out1, count, offset, it);
                }
            }
            else {
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce<T, vec_size>(out1, in1, in2, count, offset, reduction, it);
                }
            }

            s = r;
            r = (s - 1 + comm_size) % comm_size;
        }
    }
}

template <typename T, int vec_size>
struct ReduceScatterKernel {
    ReduceScatterKernel(const void *send_buf,
                        void *recv_buf,
                        size_t recv_count,
                        size_t send_stride,
                        ccl_reduction_data reduction,
                        const reduce_scatter_coll_params params)
            : send_buf(send_buf),
              recv_buf(recv_buf),
              recv_count(recv_count),
              send_stride(send_stride),
              reduction(reduction),
              params(params) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_scatter_kernel_wrap<T, vec_size>(send_buf, recv_buf, recv_count, send_stride, reduction, params);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    const void *send_buf;
    void *recv_buf;
    size_t recv_count;
    size_t send_stride;
    ccl_reduction_data reduction;
    const reduce_scatter_coll_params params;
};

template <typename T>
sycl::event reduce_scatter_large_su_ring_single_kernel(const void *send_buf,
                                                       void *recv_buf,
                                                       size_t recv_count,
                                                       size_t send_stride,
                                                       ccl::datatype dtype,
                                                       ccl::reduction reduction,
                                                       ccl_comm *comm,
                                                       std::shared_ptr<ccl_comm> p2p_comm,
                                                       ccl_stream *global_stream,
                                                       sycl_ptrs_type &sycl_ptrs,
                                                       const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    const int N = p2p_comm->size();
    const int rank = p2p_comm->rank();
    const int dest_rank = (rank + 1) % N;

    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, recv_count, dsize) &&
                           can_use_full_vector(send_buf, recv_buf, send_stride, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    std::array<void *, pipeline_size> l_remote_work_ptrs, l_work_ptrs;

    const size_t recv_bytes = recv_count * dsize;
    // the segment variables are for the multi-kernel rings; this one always takes a whole
    // tmp buffer per slot, so its chunks only ever come from a slice that does not fit
    const size_t chunk_size = get_tmp_buf_size_per_rank();

    std::array<void *, pipeline_size> work_bufs;
    std::array<void *, pipeline_size> remote_work_bufs;
    for (int i = 0; i < pipeline_size; i++) {
        work_bufs[i] = get_large_tmp_buf(i, comm);
        remote_work_bufs[i] = get_remote_large_tmp_buf(i, p2p_comm.get())[dest_rank];
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t wg_size = ccl::global_data::env().sycl_work_group_size;
    assert(wg_size > 0 && wg_size <= max_work_group_size);
    const size_t sg_size = 16;

    constexpr bool use_root_sync = true;
    const size_t max_threads = ccl::global_data::get().ze_data->devices[0].total_threads * 4; // large GRF mode
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = max_threads;
    num_threads = std::min(num_threads, max_threads);

    const size_t data_count = std::min(chunk_size / dsize, recv_count);
    const size_t kernel_threads_actual = data_count / vec_size + data_count % vec_size;
    const size_t kernel_threads = std::min(kernel_threads_actual, num_threads);
    const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
    sycl::nd_range<1> nd_range(kernel_size, wg_size);

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    reduce_scatter_coll_params params;
    params.flag = p2p_comm->flag_data();
    params.barrier_data = p2p_comm->barrier_data();
    params.work_bufs = work_bufs;
    params.remote_work_bufs = remote_work_bufs;
    params.chunk_size = chunk_size;

    if (use_full_vector) {
        ReduceScatterKernel<T, full_vec_size> rs_kernel(send_buf, recv_buf, recv_count, send_stride, reduction_op, params);
        q.parallel_for(nd_range, rs_kernel);
    }
    else {
        ReduceScatterKernel<T, partial_vec_size> rs_kernel(send_buf, recv_buf, recv_count, send_stride, reduction_op, params);
        q.parallel_for(nd_range, rs_kernel);
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, recv_count, comm->size(), dtype, evs);
    }
    else {
        std::vector<sycl::event> evts;
        work_event = submit_wait_on_events(q, evts);
    }

    return work_event;
}

template <typename T>
sycl::event reduce_scatter_large_su_ring(const void *send_buf,
                                         void *recv_buf,
                                         size_t recv_count,
                                         size_t send_stride,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         ccl_comm *comm,
                                         std::shared_ptr<ccl_comm> p2p_comm,
                                         ccl_stream *global_stream,
                                         sycl_ptrs_type &sycl_ptrs,
                                         const ccl::vector_class<ccl::event> &deps) {
    const bool use_memcpy = ccl::selection::call_env().sycl_copy_engine;
    const bool use_single_kernel = ccl::global_data::env().sycl_simple_single_kernel;

    if (use_memcpy || !use_single_kernel) {
        return reduce_scatter_large_su_ring_multi_kernels<T>(
            send_buf, recv_buf, recv_count, send_stride, dtype, reduction, comm, p2p_comm, global_stream, sycl_ptrs, deps);
    }
    else {
        return reduce_scatter_large_su_ring_single_kernel<T>(
            send_buf, recv_buf, recv_count, send_stride, dtype, reduction, comm, p2p_comm, global_stream, sycl_ptrs, deps);
    }
}

#ifndef CCL_NO_EXTERN_TEMPLATE_INST
// Suppress local instantiation in TUs that only call this impl via
// invoke_collective. Instantiations live in the sibling inst/ .cpp files.

#define TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(DT) \
    template sycl::event reduce_scatter_large_su_ring<DT>(const void *send_buf, \
                                                          void *recv_buf, \
                                                          size_t recv_count, \
                                                          size_t send_stride, \
                                                          ccl::datatype dtype, \
                                                          ccl::reduction reduction, \
                                                          ccl_comm *comm, \
                                                          std::shared_ptr<ccl_comm> p2p_comm, \
                                                          ccl_stream *global_stream, \
                                                          sycl_ptrs_type &sycl_ptrs, \
                                                          const ccl::vector_class<ccl::event> &deps)

extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(short);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(int8_t);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(uint8_t);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(int);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(uint32_t);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(int64_t);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(uint64_t);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(float);
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(double);
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(sycl::half);
#endif
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
extern TEMPLATE_REDUCE_SCATTER_LARGE_SU_RING(sycl::ext::oneapi::bfloat16);
#endif
#endif // !CCL_NO_EXTERN_TEMPLATE_INST
