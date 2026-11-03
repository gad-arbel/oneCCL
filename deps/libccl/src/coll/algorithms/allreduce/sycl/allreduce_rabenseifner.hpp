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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/host_task.hpp"

#include "proxy/proxy_types.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

namespace ccl {
namespace v1 {

template <typename T>
sycl::event allreduce_sycl_rabenseifner_blocking(sycl::queue &q,
                                                 const void *send_buf,
                                                 void *recv_buf,
                                                 size_t count,
                                                 datatype dtype,
                                                 reduction reduction,
                                                 ccl_comm *comm,
                                                 const ccl::vector_class<ccl::event> &deps,
                                                 bool &done) {
    sycl::event sycl_e;
    int newrank;

    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    int pow2 = comm->pof2();
    if (count < pow2) {
        LOG_WARN("allreduce_sycl_rabenseifner_blocking: count is too small");
        done = false;
        return sycl_e;
    }

    // blocking
    q.wait();

    // to make sure that buffer con hold the largest possible block
    // align an allocation size up to 4 bytes, to have a better vector reduction
    size_t total_size = count * ccl_dtype.size();
    void *tmp_buf = NULL;
    int to_free = 0;
    if (total_size <= comm->get_scaleout_device_buf_size()) {
        tmp_buf = comm->get_scaleout_device_buf(q);
    }
    else {
        LOG_WARN("allreduce_recursive_halving_blocking device buffer is not big enough",
                 total_size,
                 " ",
                 comm->get_scaleout_device_buf_size());
        tmp_buf = sycl::malloc_device(total_size, q);
        to_free = 1;
    }
    //fprintf(stderr, "calling allreduce_sycl_reduce_scatter_allgather_blocking count: %ld\n", count);

    if (send_buf != recv_buf) {
        sycl_e = q.submit([=](sycl::handler &h) {
            h.memcpy(recv_buf, send_buf, total_size);
        });
        sycl_e.wait();
    }

    int rem = world - pow2;

    int ep_idx = 0;
    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    // ranks are divided into two groups, first 2 * rem, and the rest.
    // In the first group, even rank send its data to odd rank, and they do
    // not participate the rabenseifner.
    // After this phase, the odd ranks of the first group and the second
    // group go to the main algorithm.
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            atl_req_t send_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, recv_buf, total_size, rank + 1, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            newrank = -1;
        }
        else {
            atl_req_t recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, tmp_buf, total_size, rank - 1, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            // tmp buffers are aligned
            bool use_full_vector = can_use_full_vector(tmp_buf, recv_buf, 1, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_pair_invoke<T, vec_size, 16>(
                    q, tmp_buf, recv_buf, recv_buf, count, reduction_op, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_pair_invoke<T, vec_size, 64>(
                    q, tmp_buf, recv_buf, recv_buf, count, reduction_op, { sycl_e });
            }
            sycl_e.wait();
            newrank = rank / 2;
        }
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        std::vector<size_t> newcnts(pow2, count / pow2);
        std::vector<size_t> newdisps(pow2);
        int re = count % pow2;
        if (re > 0) {
            for (int i = 0; i < re; i++)
                newcnts[i] += 1;
        }

        newdisps[0] = 0;
        for (int i = 1; i < pow2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        int mask = 0x1;
        int send_idx = 0, recv_idx = 0;
        int last_idx = pow2;
        while (mask < pow2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                // recv first half, send the second half
                send_idx = recv_idx + pow2 / (mask * 2);
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }
            else {
                // send first half, recv the second half
                recv_idx = send_idx + pow2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            //fprintf(stderr, "RS Rank %d, mask: %x newrank: %d dst: %d send_idx %d, recv_idx %d, last_idx %d send_cnt %ld, recv_cnt %ld\n", rank, mask, newrank, dst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt);
            atl_req_t send_req, recv_req;
            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *tbuf = (char *)tmp_buf + newdisps[recv_idx] * ccl_dtype.size();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, tbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            bool use_full_vector = can_use_full_vector(tbuf, rbuf, 4, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_pair_invoke<T, vec_size, 16>(
                    q, tbuf, rbuf, rbuf, recv_cnt, reduction_op, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_pair_invoke<T, vec_size, 64>(
                    q, tbuf, rbuf, rbuf, recv_cnt, reduction_op, { sycl_e });
            }
            sycl_e.wait();

            send_idx = recv_idx;
            mask <<= 1;

            if (mask < pow2)
                last_idx = recv_idx + pow2 / mask;
        }

        // allgather
        mask >>= 1;
        int half = 1;
        while (mask > 0) {
            int newdst = newrank ^ mask;
            /* find real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pow2 / 2)
                    last_idx = last_idx + half;

                recv_idx = send_idx + half;
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx - half;
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }

            //fprintf(stderr, "Allgather Rank %d, mask: %x newrank: %d dst: %d send_idx %d, recv_idx %d, last_idx %d, send_cnt %ld, recv_cnt %ld\n", rank, mask, newrank, dst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt);
            atl_req_t send_req, recv_req;
            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, rbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            half <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) {
            atl_req_t send_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->send(ep_idx, recv_buf, total_size, rank - 1, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
        }
        else {
            atl_req_t recv_req;
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->recv(ep_idx, recv_buf, total_size, rank + 1, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
        }
    }

    if (to_free) {
        sycl::free(tmp_buf, q);
    }
    else {
        comm->put_scaleout_device_buf(tmp_buf);
    }

    done = true;
    return sycl_e;
}

template <typename T>
sycl::event allreduce_sycl_rabenseifner_nonblocking(sycl::queue &q,
                                                    const void *send_buf,
                                                    void *recv_buf,
                                                    size_t count,
                                                    datatype dtype,
                                                    reduction reduction,
                                                    ccl_comm *comm,
                                                    const ccl::vector_class<ccl::event> &deps,
                                                    bool original_deps,
                                                    sycl_scaleout_tune_attr tune_attr,
                                                    bool &done) {
    sycl::event sycl_e;
    int newrank;

    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    // tuning parameters
    size_t pipeline_chunk_size = tune_attr.pipeline_chunk_size();

    int pow2 = comm->pof2();
    if (count < pow2) {
        LOG_WARN("allreduce_sycl_rabenseifner_nonblocking: count is too small");
        done = false;
        return sycl_e;
    }

    size_t total_size = count * ccl_dtype.size();
    void *tmp_buf = NULL;
    int to_free = 0;
    if (total_size <= comm->get_scaleout_device_buf_size()) {
        tmp_buf = comm->get_scaleout_device_buf(q);
    }
    else {
        LOG_WARN("allreduce_recursive_halving_blocking device buffer is not big enough:",
                 total_size,
                 " ",
                 comm->get_scaleout_device_buf_size());
        tmp_buf = sycl::malloc_device(total_size, q);
        to_free = 1;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (original_deps) {
        sycl_e = submit_wait_on_events(q, dep_events);
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }
    else {
        CCL_THROW_IF_NOT(dep_events.size() > 0,
                         "allreduce_sycl_rabenseifner_nonblocking: empty dependence");
    }

    if (send_buf != recv_buf) {
        sycl_e = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, total_size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }

    int rem = world - pow2;

    int ep_idx = 0;
    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);
    bool use_rdma = transport_rdma_enabled();

    // ranks are divided into two groups, first 2 * rem, and the rest.
    // In the first group, even rank send its data to odd rank, and they do
    // not participate the rabenseifner.
    // After this phase, the odd ranks of the first group and the second
    // group go to the main algorithm.
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            sycl_e = gpu_send_plain(
                q, recv_buf, count, rank + 1, tag, dtype, comm, dep_events, use_rdma);
            newrank = -1;
        }
        else {
            sycl_e =
                gpu_recv_plain(q, tmp_buf, count, rank - 1, tag, dtype, comm, dep_events, use_rdma);
            // tmp buffers are aligned
            bool use_full_vector = can_use_full_vector(tmp_buf, recv_buf, 1, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_pair_invoke<T, vec_size, 16>(
                    q, tmp_buf, recv_buf, recv_buf, count, reduction_op, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_pair_invoke<T, vec_size, 64>(
                    q, tmp_buf, recv_buf, recv_buf, count, reduction_op, { sycl_e });
            }
            newrank = rank / 2;
        }
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        sycl::queue q_worker = q;
        if (!ccl::global_data::env().sycl_enable_pipeline_gpu_rdma) {
            // out-of-order queue
            q_worker = sycl::queue(q.get_context(), q.get_device());
        }

        std::vector<size_t> newcnts(pow2, count / pow2);
        std::vector<size_t> newdisps(pow2);
        int re = count % pow2;
        if (re > 0) {
            for (int i = 0; i < re; i++)
                newcnts[i] += 1;
        }

        newdisps[0] = 0;
        for (int i = 1; i < pow2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        size_t nchunks;
        int mask = 0x1;
        int send_idx = 0, recv_idx = 0;
        int last_idx = pow2;
        while (mask < pow2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                // recv first half, send the second half
                send_idx = recv_idx + pow2 / (mask * 2);
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }
            else {
                // send first half, recv the second half
                recv_idx = send_idx + pow2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            // the partner sends what this rank receives and receives what it sends, so
            // max(send_cnt, recv_cnt) is the same on both ends; send_cnt alone is not
            pipe_prep(send_cnt > recv_cnt ? send_cnt : recv_cnt,
                      ccl_dtype.size(),
                      pipeline_chunk_size,
                      nchunks);
            LOG_DEBUG("Allreduce rabenseifner RS send count: ",
                      send_cnt,
                      " recv count: ",
                      recv_cnt,
                      " nchunks: ",
                      nchunks,
                      " pipeline_chunk_size: ",
                      pipeline_chunk_size);

            //fprintf(stderr, "Rank %d, newrank: %d dst: %d send_idx %d, recv_idx %d, send_size %ld, recv_size %ld, last_idx %d\n", rank, newrank, dst, send_idx, recv_idx, send_size, recv_size, last_idx);

            void *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            void *tbuf = (char *)tmp_buf + newdisps[recv_idx] * ccl_dtype.size();
            sycl_e = pipe_sendrecv(q_worker,
                                   sbuf,
                                   send_cnt,
                                   dst,
                                   tag,
                                   tbuf,
                                   recv_cnt,
                                   dst,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma);

            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            bool use_full_vector = can_use_full_vector(tbuf, rbuf, 4, 4);
            if (use_full_vector) {
                constexpr int vec_size = get_num_elements<T, 8, true>();
                sycl_e = reduce_pair_invoke<T, vec_size, 16>(
                    q_worker, tbuf, rbuf, rbuf, recv_cnt, reduction_op, { sycl_e });
            }
            else {
                constexpr int vec_size = get_num_elements<T, 8, false>();
                sycl_e = reduce_pair_invoke<T, vec_size, 64>(
                    q_worker, tbuf, rbuf, rbuf, recv_cnt, reduction_op, { sycl_e });
            }
            dep_events.clear();
            dep_events.push_back(std::move(sycl_e));

            send_idx = recv_idx;
            mask <<= 1;

            if (mask < pow2)
                last_idx = recv_idx + pow2 / mask;
        }

        // allgather
        mask >>= 1;
        int half = 1;
        while (mask > 0) {
            int newdst = newrank ^ mask;
            /* real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pow2 / 2)
                    last_idx = last_idx + half;

                recv_idx = send_idx + half;
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx - half;
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }

            // the partner sends what this rank receives and receives what it sends, so
            // max(send_cnt, recv_cnt) is the same on both ends; send_cnt alone is not
            pipe_prep(send_cnt > recv_cnt ? send_cnt : recv_cnt,
                      ccl_dtype.size(),
                      pipeline_chunk_size,
                      nchunks);
            LOG_DEBUG("Allreduce rabenseifner AG send count: ",
                      send_cnt,
                      " recv count: ",
                      recv_cnt,
                      " nchunks: ",
                      nchunks,
                      " pipeline_chunk_size: ",
                      pipeline_chunk_size);

            void *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            void *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            sycl_e = pipe_sendrecv(q_worker,
                                   sbuf,
                                   send_cnt,
                                   dst,
                                   tag,
                                   rbuf,
                                   recv_cnt,
                                   dst,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma);
            dep_events.clear();
            dep_events.push_back(std::move(sycl_e));

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            half <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) {
            sycl_e = gpu_send_plain(
                q, recv_buf, count, rank - 1, tag, dtype, comm, dep_events, use_rdma);
        }
        else {
            sycl_e = gpu_recv_plain(
                q, recv_buf, count, rank + 1, tag, dtype, comm, dep_events, use_rdma);
        }
        dep_events.clear();
        dep_events.push_back(std::move(sycl_e));
    }

    // submit to in-order queue
    sycl_e = submit_wait_on_events(q, dep_events);

    if (to_free) {
        sycl_e = ccl_host_task(q, {sycl_e}, [=]() {
            sycl::free(tmp_buf, q);
        });
    }
    else {
        comm->put_scaleout_device_buf(tmp_buf);
    }

    done = true;
    return sycl_e;
}

inline sycl::event allreduce_scaleout_sycl_rabenseifner(sycl::queue &q,
                                                        const void *send_buf,
                                                        void *recv_buf,
                                                        size_t count,
                                                        datatype dtype,
                                                        reduction reduction,
                                                        ccl_comm *comm,
                                                        const ccl::vector_class<ccl::event> &deps,
                                                        bool original_deps,
                                                        sycl_scaleout_tune_attr tune_attr,
                                                        bool &done) {
    auto lambda = [&]<typename T>() {
        if (ccl::global_data::env().enable_op_sync) {
            return allreduce_sycl_rabenseifner_blocking<T>(
                q, send_buf, recv_buf, count, dtype, reduction, comm, deps, done);
        }
        else {
            return allreduce_sycl_rabenseifner_nonblocking<T>(q,
                                                              send_buf,
                                                              recv_buf,
                                                              count,
                                                              dtype,
                                                              reduction,
                                                              comm,
                                                              deps,
                                                              original_deps,
                                                              tune_attr,
                                                              done);
        }
    };

    return invoke_scaleout_collective(lambda, dtype);
}

/* ======== Proxy service path ======== */

// Proxy-adapted blocking rabenseifner allreduce for use by the proxy thread.
// Same recursive halving/doubling as allreduce_sycl_rabenseifner_blocking, but the
// ATL exchange runs here on the proxy thread (staged through host buffers when GPU
// RDMA is unavailable) and every reduction is issued to the persistent kernel over
// the command ring instead of being submitted to a SYCL queue.
//
// The initial send_buf -> recv_buf copy is done by the caller on its own queue
// before the kernel is launched, so this always sees an in-place layout.
inline void allreduce_proxy_rabenseifner_blocking(ccl_proxy_context &ctx,
                                                  const void *send_buf,
                                                  void *recv_buf,
                                                  size_t count,
                                                  ccl::datatype dtype,
                                                  ccl::reduction reduction,
                                                  ccl_comm *comm,
                                                  void *aux_buffer,
                                                  size_t pipeline_chunk_size) {
    CCL_THROW_IF_NOT(send_buf == recv_buf,
                     "proxy rabenseifner allreduce expects an in-place layout");

    int newrank;

    const int world = comm->size();
    const int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool is_rdma_supported = ctx.staging->m_is_rdma_supported;
    const size_t num_work_groups = ctx.num_work_groups;

    const int pow2 = comm->pof2();
    const int rem = world - pow2;
    const size_t total_size = count * ccl_dtype.size();

    // Obtain a buffer for received blocks. A non-null aux_buffer means the caller
    // pre-allocated dedicated device memory for this op and owns its lifetime; a
    // null one means we take a slot from the communicator's scaleout
    // device-buffer pool ourselves.
    const bool from_pool = (aux_buffer == nullptr);
    if (from_pool) {
        aux_buffer = comm->get_scaleout_device_buf(*ctx.queue);
    }
    void *tmp_buf = aux_buffer;

    // tag creation
    const int ep_idx = 0;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag =
        atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    LOG_DEBUG("allreduce_proxy_rabenseifner_blocking: rank ",
              rank,
              "/",
              world,
              ", count ",
              count,
              ", pow2 ",
              pow2,
              ", rem ",
              rem,
              ", chunk_size ",
              pipeline_chunk_size);

    // ranks are divided into two groups, first 2 * rem, and the rest.
    // In the first group, even rank send its data to odd rank, and they do
    // not participate the rabenseifner.
    // After this phase, the odd ranks of the first group and the second
    // group go to the main algorithm.
    if (rank < 2 * rem) {
        if (rank % 2 == 0) {
            if (is_rdma_supported) {
                atl_req_t send_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->send(ep_idx, recv_buf, total_size, rank + 1, tag, send_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            }
            else {
                ccl_proxy_staging::do_send(
                    ctx.staging, recv_buf, total_size, rank + 1, pipeline_chunk_size, comm, tag);
            }
            newrank = -1;
        }
        else {
            if (is_rdma_supported) {
                atl_req_t recv_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->recv(ep_idx, tmp_buf, total_size, rank - 1, tag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
            else {
                ccl_proxy_staging::do_recv(
                    ctx.staging, tmp_buf, total_size, rank - 1, pipeline_chunk_size, comm, tag);
            }

            // This is the largest reduce the algorithm issues: the whole count
            ctx.producer->submit_and_wait(ccl_ring_cmd_type::reduce,
                                          tmp_buf,
                                          recv_buf,
                                          recv_buf,
                                          static_cast<uint32_t>(count),
                                          dtype,
                                          num_work_groups);
            newrank = rank / 2;
        }
    }
    else {
        newrank = rank - rem;
    }

    if (newrank != -1) {
        std::vector<size_t> newcnts(pow2, count / pow2);
        std::vector<size_t> newdisps(pow2);
        int re = count % pow2;
        if (re > 0) {
            for (int i = 0; i < re; i++)
                newcnts[i] += 1;
        }

        newdisps[0] = 0;
        for (int i = 1; i < pow2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        int mask = 0x1;
        int send_idx = 0, recv_idx = 0;
        int last_idx = pow2;
        while (mask < pow2) {
            int newdst = newrank ^ mask;
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                // recv first half, send the second half
                send_idx = recv_idx + pow2 / (mask * 2);
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }
            else {
                // send first half, recv the second half
                recv_idx = send_idx + pow2 / (mask * 2);
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *tbuf = (char *)tmp_buf + newdisps[recv_idx] * ccl_dtype.size();
            if (is_rdma_supported) {
                atl_req_t send_req, recv_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->recv(ep_idx, tbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
            else {
                ccl_proxy_staging::do_sendrecv(ctx.staging,
                                               sbuf,
                                               send_cnt * ccl_dtype.size(),
                                               dst,
                                               tbuf,
                                               recv_cnt * ccl_dtype.size(),
                                               dst,
                                               pipeline_chunk_size,
                                               comm,
                                               tag);
            }

            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            ctx.producer->submit_and_wait(ccl_ring_cmd_type::reduce,
                                          tbuf,
                                          rbuf,
                                          rbuf,
                                          static_cast<uint32_t>(recv_cnt),
                                          dtype,
                                          num_work_groups);

            send_idx = recv_idx;
            mask <<= 1;

            if (mask < pow2)
                last_idx = recv_idx + pow2 / mask;
        }

        // allgather
        mask >>= 1;
        int half = 1;
        while (mask > 0) {
            int newdst = newrank ^ mask;
            /* find real rank of dest */
            int dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            size_t send_cnt = 0, recv_cnt = 0;
            if (newrank < newdst) {
                /* update last_idx except on first iteration */
                if (mask != pow2 / 2)
                    last_idx = last_idx + half;

                recv_idx = send_idx + half;
                for (int i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (int i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }
            else {
                recv_idx = send_idx - half;
                for (int i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
                for (int i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
            }

            char *sbuf = (char *)recv_buf + newdisps[send_idx] * ccl_dtype.size();
            char *rbuf = (char *)recv_buf + newdisps[recv_idx] * ccl_dtype.size();
            if (is_rdma_supported) {
                atl_req_t send_req, recv_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->send(ep_idx, sbuf, send_cnt * ccl_dtype.size(), dst, tag, send_req));
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->recv(ep_idx, rbuf, recv_cnt * ccl_dtype.size(), dst, tag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
            else {
                ccl_proxy_staging::do_sendrecv(ctx.staging,
                                               sbuf,
                                               send_cnt * ccl_dtype.size(),
                                               dst,
                                               rbuf,
                                               recv_cnt * ccl_dtype.size(),
                                               dst,
                                               pipeline_chunk_size,
                                               comm,
                                               tag);
            }

            if (newrank > newdst)
                send_idx = recv_idx;

            mask >>= 1;
            half <<= 1;
        }
    }

    if (rank < 2 * rem) {
        if (rank % 2) {
            if (is_rdma_supported) {
                atl_req_t send_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->send(ep_idx, recv_buf, total_size, rank - 1, tag, send_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
            }
            else {
                ccl_proxy_staging::do_send(
                    ctx.staging, recv_buf, total_size, rank - 1, pipeline_chunk_size, comm, tag);
            }
        }
        else {
            if (is_rdma_supported) {
                atl_req_t recv_req;
                ATL_CALL_THROW_IF_ERROR(
                    atl_comm->recv(ep_idx, recv_buf, total_size, rank + 1, tag, recv_req));
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            }
            else {
                ccl_proxy_staging::do_recv(
                    ctx.staging, recv_buf, total_size, rank + 1, pipeline_chunk_size, comm, tag);
            }
        }
    }

    // Release the pool slot we acquired above.
    // A caller-preallocated aux_buffer is left for the caller to free.
    if (from_pool) {
        comm->put_scaleout_device_buf(tmp_buf);
    }
}

// Proxy service wrapper for rabenseifner allreduce
// Unpacks ccl_proxy_op and calls allreduce_proxy_rabenseifner_blocking
inline void proxy_allreduce_rabenseifner(const ccl_proxy_op &op, ccl_proxy_context &ctx) {
    // Helper to get pipeline chunk size
    const size_t chunk_size = op.pipeline_chunk_size
                                  ? op.pipeline_chunk_size
                                  : ccl::global_data::env().sycl_max_pipeline_chunk_size;

    // Unpack operation descriptor and call the algorithm
    allreduce_proxy_rabenseifner_blocking(
        ctx, op.src, op.dst, op.count, op.dtype, op.reduction, op.comm, op.aux_dev, chunk_size);
}

// Persistent-kernel based rabenseifner allreduce scale-out algorithm
inline sycl::event allreduce_scaleout_sycl_proxy_rabenseifner(
    sycl::queue &q,
    const void *send_buf,
    void *recv_buf,
    size_t count,
    datatype dtype,
    reduction reduction,
    ccl_comm *comm,
    const ccl::vector_class<ccl::event> &deps,
    sycl_scaleout_tune_attr tune_attr,
    bool &done) {
    sycl::event scaleout_event;

    const bool is_recording = use_recording_path(q);
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const int rank = comm->rank();
    const int world = comm->size();
    const int pow2 = comm->pof2();
    const int rem = world - pow2;

    // The algorithm splits count into pow2 blocks, so it cannot run when there
    // are fewer elements than blocks. Decline before launching anything, the
    // caller then falls back to another scale-out algorithm.
    if (count < static_cast<size_t>(pow2)) {
        LOG_WARN("allreduce_scaleout_sycl_proxy_rabenseifner: count is too small");
        done = false;
        return scaleout_event;
    }

    const size_t total_size = count * ccl_dtype.size();

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // The proxy algorithm expects an in-place layout: do the send_buf -> recv_buf
    // copy here on the caller's queue (chained on the input deps) and point the
    // proxy op's source at recv_buf.
    if (send_buf != recv_buf) {
        sycl::event copy_event = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, total_size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_event));
    }

    // Decide how the buffer for received blocks is provided. Unlike the ring, this
    // algorithm receives the whole count in the first-group phase, so a pool slot
    // is only usable when it can hold total_size; otherwise allocate a dedicated
    // device buffer here
    bool to_free = false;
    void *aux_buffer = nullptr;
    if (total_size > comm->get_scaleout_device_buf_size()) {
        CCL_THROW_IF_NOT(!is_recording, "large-buffer allocation does not support recording");
        aux_buffer = sycl::malloc_device(total_size, q);
        to_free = true;
    }

    // Decide work-group count based on the smallest per-iteration reduce block
    const size_t min_block_size = (count / pow2) * ccl_dtype.size();
    const size_t num_work_groups = select_persistent_work_group_num(min_block_size);

    LOG_DEBUG("allreduce_scaleout_sycl_proxy_rabenseifner: rank ",
              rank,
              "/",
              world,
              ", count ",
              count,
              ", pow2 ",
              pow2,
              ", rem ",
              rem,
              ", num_work_groups ",
              num_work_groups);

    ccl_proxy_op op{ ccl_proxy_op_type::allreduce_rabenseifner,
                     recv_buf, // in-place after the copy above
                     recv_buf,
                     count,
                     dtype,
                     reduction,
                     comm,
                     /*aux=*/aux_buffer };
    // Optional
    op.pipeline_chunk_size = tune_attr.pipeline_chunk_size();
    op.num_work_groups = num_work_groups;
    // The even ranks of the first group only forward their data and receive the
    // result, they issue no ring commands at all, so a sync-only kernel is enough.
    op.sync_only = (rank < 2 * rem && rank % 2 == 0);

    scaleout_event = submit_and_launch_persistent_kernel(q, op, dep_events);

    // Free the large-buffer fallback allocation once the kernel has finished.
    if (to_free) {
        scaleout_event = ccl_host_task(q, {scaleout_event}, [=]() {
            sycl::free(aux_buffer, q);
        });
    }

    done = true;
    return scaleout_event;
}

} // namespace v1
} // namespace ccl
