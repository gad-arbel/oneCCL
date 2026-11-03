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

#define ALGO_SELECTION_ENV "CCL_GATHER"

#include "test_impl.hpp"

template <typename T>
class gather_test : public base_test<T> {
public:
    /*
     * Gather semantic verification:
     * - Every rank owns send_buf of count elements.
     * - Root receives nranks * count elements; slot r = recv_buf + r * count.
     * - Non-root recv_buf is unused.
     *
     * In-place condition (root only):
     *   send_buf == recv_buf + root * count * dtype_size
     */
    int check(test_operation<T>& op) {
        /* only root's recv_buf is meaningful */
        if (op.comm_rank != ROOT_RANK)
            return TEST_SUCCESS;

        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (int r = 0; r < op.comm_size; r++) {
                for (size_t elem_idx = 0; elem_idx < op.elem_count;
                     elem_idx += op.get_check_step(elem_idx)) {
                    /* each rank sends: rank * elem_count + elem_idx (% 256) */
                    T expected = static_cast<T>((r * op.elem_count + elem_idx) % 256);
                    size_t flat_idx = static_cast<size_t>(r) * op.elem_count + elem_idx;
                    if (base_test<T>::check_error(op, expected, buf_idx, flat_idx))
                        return TEST_FAILURE;
                }
            }
        }
        return TEST_SUCCESS;
    }

    /* every rank fills its own send buffer */
    void fill_send_buffers(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < op.elem_count; elem_idx++) {
                op.send_bufs[buf_idx][elem_idx] =
                    static_cast<T>((op.comm_rank * op.elem_count + elem_idx) % 256);
            }
        }
    }

    /* For PLACE_IN the framework uploads the host recv buffer into
     * device_recv_bufs, not device_send_bufs (see
     * base_test::copy_to_device_send_buffers), so what a rank will send has to be
     * staged in its recv buffer. Seed block comm_rank, mirroring allgather_test:
     * on root that block is where the in-place condition
     * send_buf == recv_buf + root * count * dtype_size puts send_buf. */
    void fill_recv_buffers(test_operation<T>& op) {
        if (op.param.place_type != PLACE_IN) {
            base_test<T>::fill_recv_buffers(op);
            return;
        }

        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < op.elem_count; elem_idx++) {
                op.recv_bufs[buf_idx][op.comm_rank * op.elem_count + elem_idx] =
                    static_cast<T>((op.comm_rank * op.elem_count + elem_idx) % 256);
            }
        }
    }

    void run_derived(test_operation<T>& op) {
        void* send_buf;
        void* recv_buf;
        auto param = op.get_param();
        auto attr = ccl::create_operation_attr<ccl::gather_attr>();

        for (auto buf_idx : op.buf_indexes) {
            op.prepare_attr(attr, buf_idx);

            /* In-place sends out of block comm_rank, where fill_recv_buffers
             * staged it. Spelling out the offset keeps the in-place condition
             * exact for any ROOT_RANK, not just rank 0. */
            if (param.place_type == PLACE_IN) {
                send_buf = static_cast<char*>(op.get_recv_buf(buf_idx)) +
                           op.comm_rank * op.elem_count * op.datatype_size;
            }
            else {
                send_buf = op.get_send_buf(buf_idx);
            }
            /* only root supplies a recv_buf */
            recv_buf = (op.comm_rank == ROOT_RANK) ? op.get_recv_buf(buf_idx) : nullptr;

            op.events.push_back(ccl::gather(send_buf,
                                            recv_buf,
                                            op.elem_count,
                                            op.datatype,
                                            ROOT_RANK,
                                            transport_data::instance().get_comm(),
                                            transport_data::instance().get_stream(),
                                            attr));
        }
    }
};

RUN_METHOD_DEFINITION(gather_test);
TEST_CASES_DEFINITION(gather_test);
MAIN_FUNCTION_WITH_TRY CATCH_EXCEPTION(ccl::exception) CATCH_EXCEPTION(CCL_TEST_SYCL_EXCEPTION);
