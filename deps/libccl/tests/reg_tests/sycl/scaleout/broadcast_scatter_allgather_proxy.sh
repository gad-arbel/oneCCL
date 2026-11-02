#!/bin/bash

# Exercise the broadcast Scatter-AllGather proxy path.
#
# Shape: 2 nodes x 2 ranks/node = 4 ranks total, -ppn 2.
# Force mode prevents a silent fallback to the legacy broadcast path.
# The data check proves correctness, and the markers prove that every rank
# executed the rail broadcast through the proxy thread.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename "$0" .sh)"
TEST_LOG="${BASENAME}.log"

source "${ROOT_DIR}/utils.sh"

make_common_actions "${SCRIPT_DIR}" "${TEST_LOG}" "sycl"

export I_MPI_JOB_TIMEOUT=120

cmd="CCL_SYCL_FORCE_PCIE=1"
cmd+=" CCL_SYCL_P2P_LEVEL=node"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_ENABLE=1"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_FORCE=1"
cmd+=" CCL_SYCL_BROADCAST_SCATTER_ALLGATHER_THRESHOLD=0"
cmd+=" CCL_SYCL_ENABLE_BROADCAST_SCALEOUT_PROXY=1"
cmd+=" CCL_LOG_LEVEL=info"
cmd+=" mpiexec -l -n 4 -ppn 2 ${SCRIPT_DIR}/benchmark"
cmd+=" -a gpu -m usm -u device -l broadcast"
cmd+=" -i 2 -w 1 -y 4096"
cmd+=" -j off --check last -d int32 -e in_order"
cmd+=" > ${TEST_LOG} 2>&1"

echo "  |TEST CMD| ${cmd}"
run_cmd "${cmd}"
check_log "${TEST_LOG}"

expected_proxy_markers=4
actual_proxy_markers=$(grep -c "BCAST_SAG_PROXY_THREAD_ACTIVE" "${TEST_LOG}" || true)
if [ "${actual_proxy_markers}" -ne "${expected_proxy_markers}" ]; then
    echo "Error: expected ${expected_proxy_markers} proxy markers, found ${actual_proxy_markers}"
    cat "${TEST_LOG}"
    exit 1
fi

if grep -q "BCAST_SAG_PROXY_THREAD_INACTIVE" "${TEST_LOG}"; then
    echo "Error: broadcast proxy service was inactive"
    cat "${TEST_LOG}"
    exit 1
fi

if ! grep -q "BCAST_SAG_ACTIVE:.*proxy_thread" "${TEST_LOG}"; then
    echo "Error: broadcast SAG did not report the proxy rail path"
    cat "${TEST_LOG}"
    exit 1
fi

rm "${TEST_LOG}"
echo "Pass"
