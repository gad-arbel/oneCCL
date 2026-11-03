#!/bin/bash

# This test is checking if oneCCL is able to run without 
# any sycl devices available on the system to run cpu
# comunicators.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
# The cpu reg test, built by every reg-test build. The examples install tree it
# used to be taken from only exists in a standalone v1 build.
BINFILE="${ROOT_DIR}/cpu/allreduce"

source "${ROOT_DIR}/utils.sh"

make_common_actions "${SCRIPT_DIR}" "${TEST_LOG}" "sycl"

if [ ! -x "${BINFILE}" ]
then
    echo "Error: ${BINFILE} not found, run the reg-test build stage first"
    echo "Fail"
    exit 1
fi

function run_cmd() {
    if ! "$@"; then
        echo "run_cmd: $*"
        echo "Fail"
        exit 1
    fi
}

true > "${TEST_LOG}"

# We are using ONEAPI_DEVICE_SELECTOR="cuda:*", because our systems do not have 
# GPU's supporting cuda. This way we can ensure that no device is detected by
# sycl runtime.
run_cmd env ONEAPI_DEVICE_SELECTOR="cuda:*" mpiexec -n 2 "${BINFILE}"  >> "${TEST_LOG}"

check_log "${TEST_LOG}"
rm "${TEST_LOG}"
echo "Pass"
