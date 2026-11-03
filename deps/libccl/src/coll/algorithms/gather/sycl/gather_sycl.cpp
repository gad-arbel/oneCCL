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

// SYCL gather dispatchers; see gather_sycl.hpp for the entry-point roles and the
// buffer contract. Paths that cannot serve a call report done=false rather than
// throwing, handing it back to the scheduler-based algorithm.

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/p2p_domain.hpp"
#include "coll/algorithms/gather/sycl/gather_sycl.hpp"
#include "coll/algorithms/gather/sycl/gather_small_sycl.hpp"
#include "coll/algorithms/gather/sycl/gather_large_sycl.hpp"

namespace ccl {
namespace v1 {

static bool do_fallback_to_scheduler(size_t bytes) {
    const size_t threshold = ccl::global_data::env().sycl_gather_scaleout_threshold;
    return bytes > threshold;
}

// Hard capacity requirement on top of the small-size threshold: the small kernel
// pushes all comm_size blocks into root's buffer.
static bool small_kernel_fits_tmp_buf(const std::shared_ptr<ccl_comm>& p2p_comm,
                                      size_t count,
                                      size_t dsize) {
    const size_t staged_bytes = static_cast<size_t>(p2p_comm->size()) * count * dsize;
    return staged_bytes <= p2p_comm->get_tmp_buf_size();
}

event gather_scaleup_sycl(sycl::queue& q,
                          const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          datatype dtype,
                          int root,
                          ccl_comm* comm,
                          std::shared_ptr<ccl_comm> p2p_comm,
                          ccl_stream* global_stream,
                          const vector_class<event>& deps,
                          bool& done) {
    event e;
    done = true;

    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const uint32_t world = p2p_comm->size();

    if (world == 1) {
        sycl::event sycl_e;
        auto sycl_q = global_stream->get_native_stream();
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        // single rank: recv_buf holds just this rank's block, so it is a copy
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: gather");
            sycl_e = sycl_q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * dsize);
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: gather");
            sycl_e = submit_wait_on_events(sycl_q, dep_events);
        }
        return event::create_from_native(sycl_e);
    }

    const size_t bytes = count * dsize;
    if (do_fallback_to_scheduler(bytes)) {
        LOG_DEBUG("gather count size = ",
                  bytes,
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_gather_scaleout_threshold,
                  " -- falling back");
        done = false;
        return e;
    }

    const bool use_small = bytes <= ccl::global_data::env().sycl_gather_small_threshold &&
                           small_kernel_fits_tmp_buf(p2p_comm, count, dsize);
    if (use_small) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("gather_small", "send_size", bytes, comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| gather selects small kernel, count: ", count, " datatype: ", dtype);
        e = gather_small(send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| gather selects small kernel, count: ",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("gather_large", "send_size", bytes, comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| gather selects large kernel, count: ", count, " datatype: ", dtype);
        e = gather_large(send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| gather selects large kernel, count: ",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

event gather_sycl_multi_domain(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* global_comm,
                               std::shared_ptr<ccl_comm> p2p_comm,
                               std::shared_ptr<ccl_comm> r2r_comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done) {
    LOG_DEBUG("gather_sycl multi-domain count: ",
              count,
              ", r2r size: ",
              r2r_comm->size(),
              " -- no sycl scaleout engine, falling back");
    done = false;
    event e;
    return e;
}

event gather_sycl(sycl::queue q,
                  const void* send_buf,
                  void* recv_buf,
                  size_t count,
                  datatype dtype,
                  int root,
                  ccl_comm* global_comm,
                  ccl_stream* global_stream,
                  const gather_attr& attr,
                  const vector_class<event>& deps,
                  bool& done) {
    // ccl_coll_build_gather does this for the scheduler path, which this route
    // bypasses; the kernels index remote_ptrs[root] unchecked on every rank
    CCL_THROW_IF_NOT(root >= 0 && root < global_comm->size(),
                     "wrong root: ",
                     root,
                     ", comm_size: ",
                     global_comm->size());

    if (count == 0) {
        done = true;
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return event::create_from_native(e);
    }

    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    // gather has no tuned domain policy of its own, so take the default one
    auto p2p_level = ccl_p2p_domain::sycl_get_default_p2p_level(
        global_comm, count * ccl_dtype.size(), ccl::global_data::env().sycl_p2p_level_threshold);
    auto [ p2p_comm, r2r_comm ] = ccl_p2p_domain::get_p2p_comms(global_comm, p2p_level);

    // single p2p domain
    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return gather_scaleup_sycl(q,
                                   send_buf,
                                   recv_buf,
                                   count,
                                   dtype,
                                   root,
                                   global_comm,
                                   p2p_comm,
                                   global_stream,
                                   deps,
                                   done);
    }
    return gather_sycl_multi_domain(q,
                                    send_buf,
                                    recv_buf,
                                    count,
                                    dtype,
                                    root,
                                    global_comm,
                                    p2p_comm,
                                    r2r_comm,
                                    global_stream,
                                    deps,
                                    done);
}

} // namespace v1
} // namespace ccl
