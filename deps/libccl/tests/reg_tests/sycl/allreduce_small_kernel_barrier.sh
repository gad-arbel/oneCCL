#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../.." && pwd -P)
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
BINFILE="${BASENAME}"

# check_log lives here. This test ships its own binary, so it does not need
# make_common_actions, which only exists to stage the shared benchmark.
source ${ROOT_DIR}/utils.sh

# Some tests emit ANSI escape codes, but we do not want them inside the log.
export NO_COLOR=1

cd ${SCRIPT_DIR}

# CI sources vars.sh before running. For a local build tree, pick up the
# generated env.sh so libccl.so is found.
if [[ -z "${CCL_ROOT}" ]]
then
    for candidate in "${REPO_ROOT}/build/env.sh" "${REPO_ROOT}/build_release/env.sh"
    do
        if [[ -r "${candidate}" ]]
        then
            # shellcheck disable=SC1090
            source "${candidate}"
            break
        fi
    done
fi

if [[ ! -x "${SCRIPT_DIR}/${BINFILE}" ]]
then
    echo "Fail"
    echo "DEBUG: ${BINFILE} was not built."
    echo "DEBUG: build it with ./scripts/dev_build.sh -r --reg-tests -t ${BINFILE}"
    exit 1
fi

# Force the exact fused small-allreduce path that failed in graph_integration
# during its ordinary warm-up, before graph recording started.
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=0
export CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD=1048576000
export CCL_SYCL_KERNEL_SYNC=1
export CCL_RUNTIME_LIST="${CCL_RUNTIME_LIST:-level_zero:gpu}"

# 4 ranks on a single node keeps the normal PVC tile layout used by CI, which is
# what spreads the failing element across several workgroups.
mpiexec -l -n 4 -ppn 4 ${SCRIPT_DIR}/${BINFILE} > ${TEST_LOG} 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]
then
    echo "Fail"
    echo "DEBUG: ${BASENAME}.sh fails for mpiexec"
    exit 1
fi

check_log ${TEST_LOG}

rm -f ${TEST_LOG}
echo "Pass"
