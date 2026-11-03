#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

kvs_modes="pmi occp"
backends="sycl host"
transports="ofi mpi"

bench_options="-w 0 -i 1 -d int32 -l allreduce -y 1024"

for kvs_mode in ${kvs_modes}
do
    for transport in ${transports}
    do
        for backend in ${backends}
        do
            cmd="CCL_ATL_TRANSPORT=${transport} CCL_KVS_MODE=${kvs_mode}"
            cmd+=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark"
            cmd+=" ${bench_options} -b ${backend} > ${TEST_LOG} 2>&1"
            run_cmd "${cmd}"
            check_log ${TEST_LOG}
        done
    done
done

# occp needs the ofi transport, so on any other transport the library falls back
# to pmi. The occp+mpi cells above cannot tell that fallback apart from occp
# running over mpi, so assert the fallback explicitly.
DOWNGRADE_LOG="${BASENAME}_downgrade.log"

cmd="CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=occp CCL_LOG_LEVEL=info"
cmd+=" mpiexec -l -n 2 ${SCRIPT_DIR}/benchmark"
cmd+=" ${bench_options} -b host > ${DOWNGRADE_LOG} 2>&1"
run_cmd "${cmd}"

for pattern in "switch to pmi kvs mode" "CCL_KVS_MODE: pmi"
do
    if ! grep -q "${pattern}" ${DOWNGRADE_LOG}
    then
        echo "Error: expected occp to fall back to pmi on the mpi transport,"
        echo "       '${pattern}' not found in ${DOWNGRADE_LOG}"
        echo "Fail"
        exit 1
    fi
done

rm ${DOWNGRADE_LOG}
rm ${TEST_LOG}
echo "Pass"
