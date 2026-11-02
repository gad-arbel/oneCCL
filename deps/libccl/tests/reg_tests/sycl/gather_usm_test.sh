#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_counts="2 4"
# 1024 (4 KB) hits the small kernel, 262144 (1 MB) hits the large kernel.
counts="1024 262144"
# Large-kernel data path: 0 = IPC-direct, 1 = tmp-buf staged.
large_tmp_buf_modes="0 1"
# Large-kernel IPC protocol: write pushes into root, read pulls from the peers.
large_protocols="write read"
# Non-root recv_buf is ignored by the API: pass nullptr, or plain heap memory.
nonroot_recvs="null host"

for proc_count in ${proc_counts}
do
    for count in ${counts}
    do
        for large_tmp_buf_mode in ${large_tmp_buf_modes}
        do
            for large_protocol in ${large_protocols}
            do
                for nonroot_recv in ${nonroot_recvs}
                do
                    cmd="CCL_SYCL_GATHER_LARGE_TMP_BUF=${large_tmp_buf_mode}"
                    cmd+=" CCL_SYCL_GATHER_LARGE_PROTOCOL=${large_protocol}"
                    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/${BINFILE} gpu device ${nonroot_recv}"
                    cmd+=" --count ${count} > ${TEST_LOG} 2>&1"
                    run_cmd "${cmd}"
                    check_log ${TEST_LOG}
                done
            done
        done
    done
done

rm ${TEST_LOG}
echo "Pass"
