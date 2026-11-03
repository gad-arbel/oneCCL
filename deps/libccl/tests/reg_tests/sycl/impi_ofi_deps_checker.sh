#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

# Only the package's own setvars.sh may provide the environment checked below,
# so drop what the caller set up once the install under test is known.
ccl_install_dir="${CCL_ROOT}"
unset CCL_ROOT
unset I_MPI_ROOT
unset FI_PROVIDER_PATH

source "${ccl_install_dir}/env/setvars.sh"
check_ccl
check_impi

echo "CCL_ROOT: " $CCL_ROOT >> ${TEST_LOG}
echo "I_MPI_ROOT: " $I_MPI_ROOT >> ${TEST_LOG}
echo "FI_PROVIDER_PATH: " $FI_PROVIDER_PATH >> ${TEST_LOG}

proc_counts="4"
transports="ofi mpi"
ofi_provs="$(get_default_prov)"
for proc_count in ${proc_counts}
do
    for transport in ${transports}
    do
        for ofi_prov in ${ofi_provs}
        do
            cmd=" CCL_ATL_TRANSPORT=${transport}"
            cmd+=" FI_PROVIDER=${ofi_prov}"
            cmd+=" mpiexec -l -n ${proc_count} -ppn 2"
            cmd+=" ${SCRIPT_DIR}/benchmark >> ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

rm ${TEST_LOG}

echo "Pass"
