#!/bin/bash
#
# Reg-test for the CRI second-engine self copy of out-of-place allgatherv, alltoall and
# alltoallv (CCL_SYCL_<COLL>_SELF_COPY_THRESHOLD, default 8 MB).
#
# Each case runs the benchmark with value checks on every iteration, with the threshold at
# its default, 0 or 1 and block sizes around the threshold, and checks the self-copy decision
# that get_self_copy_queue() logs at debug level: parallel_ce, or the expected fallback reason.
# Only CRI makes the decision, so the test skips itself on other devices.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
APP_LOG="${BASENAME}_app.log"

source ${ROOT_DIR}/utils.sh

# CRI device ids are 0x674x and 0x675x
if ! cat /sys/class/drm/renderD*/device/device 2>/dev/null | grep -qiE "^0x67[45]"
then
    echo "No CRI device, skip"
    exit 0
fi

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

export CCL_ZE_AUTO_TUNE_PORTS=0
export CCL_LOG_LEVEL=debug

# allgatherv takes the copy engine ring only with CCL_SYCL_COPY_ENGINE=1, alltoall(v) by default
copy_engine_env="CCL_SYCL_COPY_ENGINE=1"

# block counts of int32, the default threshold is 8 MB = 2097152 elements
# 64 bytes below the threshold: still a 64-byte multiple, so allgatherv keeps its copy-engine
# ring, and above the 4 MB where CRI alltoall switches to the copy engine
count_below=2097136
count_at=2097152
count_chunked=4194304

all_colls="allgatherv,alltoall,alltoallv"
threshold_vars="CCL_SYCL_ALLGATHERV_SELF_COPY_THRESHOLD CCL_SYCL_ALLTOALL_SELF_COPY_THRESHOLD"
threshold_vars+=" CCL_SYCL_ALLTOALLV_SELF_COPY_THRESHOLD"

bench_options="-w 1 -i 3 -j off -c all -b sycl $(get_default_bench_dtype)"
proc_counts="2"

check_self_copy_path() {
    coll=$1
    expected=$2

    log_name=${coll}
    if [[ ${coll} == "allgatherv" ]]
    then
        log_name="allgather"
    fi
    decisions=$(grep -E "collective=${log_name} selected_path=" ${TEST_LOG})
    if [[ "${decisions}" == "" ]]
    then
        echo "Error: no self-copy decision for ${coll} in ${TEST_LOG}"
        exit 1
    fi
    if [[ ${expected} == "parallel_ce" ]]
    then
        pattern="selected_path=parallel_ce .* fallback_reason=none "
    else
        pattern="selected_path=serial_ce .* fallback_reason=${expected} "
    fi
    unexpected=$(echo "${decisions}" | grep -Ev "${pattern}")
    if [[ "${unexpected}" != "" ]]
    then
        echo "Error: expected ${expected} for ${coll} in ${TEST_LOG}"
        echo "${unexpected}" | head -4
        exit 1
    fi
}

# run_case <proc_count> <colls> <count> <expected> [env]
run_case() {
    proc_count=$1
    colls=$2
    count=$3
    expected=$4
    extra_env=${5:-}

    cmd="${copy_engine_env} ${extra_env}"
    cmd+=" mpiexec -l -n ${proc_count} ${SCRIPT_DIR}/benchmark ${bench_options}"
    cmd+=" -l ${colls} -y ${count} > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"

    # libccl debug lines are not errors even when they say so
    grep -v "|CCL_DEBUG|" ${TEST_LOG} > ${APP_LOG}
    check_log ${APP_LOG}
    for coll in ${colls//,/ }
    do
        check_self_copy_path ${coll} ${expected}
    done
}

for proc_count in ${proc_counts}
do
    run_case ${proc_count} ${all_colls} ${count_below} small_message
    run_case ${proc_count} ${all_colls} ${count_at} parallel_ce

    disabled_env=""
    always_env=""
    for var in ${threshold_vars}
    do
        disabled_env+=" ${var}=0"
        always_env+=" ${var}=1"
    done
    run_case ${proc_count} ${all_colls} ${count_at} disabled "${disabled_env}"
    run_case ${proc_count} ${all_colls} ${count_below} parallel_ce "${always_env}"

    run_case ${proc_count} allgatherv ${count_at} tmp_buf_path "CCL_SYCL_ALLGATHERV_TMP_BUF=1"
    # the self copy covers the whole block while the collective runs chunk by chunk
    run_case ${proc_count} allgatherv,alltoall ${count_chunked} parallel_ce \
        "CCL_SYCL_ALLGATHERV_CHUNKING_THRESHOLD=4194304 CCL_SYCL_ALLTOALL_CHUNKING_THRESHOLD=4194304"
done

rm ${TEST_LOG} ${APP_LOG}
echo "Pass"
