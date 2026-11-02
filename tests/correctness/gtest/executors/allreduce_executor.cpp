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

#include "allreduce_executor.h"

#include "gtest/gtest.h"

namespace oneccl_test {

namespace {

// Maps PatternOp to onecclRedOp_t. PRE_MUL_SUM has no mapping: onecclPreMulSum
// is not a plain constant. Building it needs a live communicator to allocate a
// handle at run time, which a static lookup cannot provide. It fails loudly
// until the op axis covers it. No default case, so -Wswitch names a new op.
bool to_reduction_op(PatternOp op, onecclRedOp_t *reduction_op) {
    switch (op) {
    case PatternOp::ONECCL_SUM:
        *reduction_op = onecclSum;
        return true;
    case PatternOp::ONECCL_PROD:
        *reduction_op = onecclProd;
        return true;
    case PatternOp::ONECCL_MAX:
        *reduction_op = onecclMax;
        return true;
    case PatternOp::ONECCL_MIN:
        *reduction_op = onecclMin;
        return true;
    case PatternOp::ONECCL_AVG:
        *reduction_op = onecclAvg;
        return true;
    case PatternOp::ONECCL_PRE_MUL_SUM:
        return false;
    }
    return false;
}

} // namespace

AllReduceExecutor::AllReduceExecutor(const CollectiveParams &params,
                                     ThreadContext &ctx)
    : CollectiveExecutor(params, ctx), count_(params.count) {}

void AllReduceExecutor::setup() {
    const DataSpec spec = params_.to_data_spec(ctx_.rank(), ctx_.comm_ranks(),
                                               /*buffer_index=*/0);
    data_ = ITestData::create(spec);

    const size_t bytes = count_ * dtype_size(params_.dtype);

    // AllReduce in-place is the degenerate case: one buffer, send == recv.
    if (params_.is_in_place) {
        // A repeated setup() refills the buffer rather than reallocating it,
        // so a freed address is not reused while a peer holds a cached handle.
        // The caller orders the refill after every rank's previous collective.
        if (recv_buff_ == nullptr) {
            recv_buff_ = make_buffer(bytes);
        }
        HostBuffer host(bytes);
        data_->fill_send(host);
        recv_buff_->h2d_copy(host);
        // No poison pair on an in-place row: the region the collective writes
        // is the region it read its inputs from.
        return;
    }

    // Allocate send and receive buffers for the out-of-place case.
    send_buff_ = make_buffer(bytes);
    recv_buff_ = make_buffer(bytes);

    // Prepare host-side buffers for data transfer.
    HostBuffer send_host(bytes);
    data_->fill_send(send_host);
    send_buff_->h2d_copy(send_host);

    HostBuffer recv_host(bytes);
    data_->poison_recv(recv_host);
    recv_buff_->h2d_copy(recv_host);
}

onecclResult_t AllReduceExecutor::execute(onecclComm_t comm) {
    onecclRedOp_t reduction_op = onecclSum;
    if (!to_reduction_op(params_.op, &reduction_op)) {
        ADD_FAILURE() << "PatternOp " << static_cast<int>(params_.op)
                      << " has no onecclRedOp_t mapping yet";
        return onecclInvalidArgument;
    }
    void *send_ptr =
        params_.is_in_place ? recv_buff_->ptr() : send_buff_->ptr();
    return onecclAllReduce(send_ptr, recv_buff_->ptr(), count_, params_.dtype,
                           reduction_op, comm, ctx_.stream());
}

SourceMap AllReduceExecutor::source_map() const {
    SourceMap map;
    map.contributors = Contributors::ONECCL_ALL_RANKS;
    return map;
}

} // namespace oneccl_test
