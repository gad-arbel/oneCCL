#!/bin/bash

# OCCP bootstrap across nodes. This is the configuration OCCP exists for: every
# rank reports to one server, which answers through one forwarder per node,
# instead of exchanging addresses rank by rank. A single-node run cannot tell a
# working waterfall from a broken one, so this test also asserts that the server
# saw exactly one forwarder per node in the allocation.

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")/../"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"
EVIDENCE_LOG="${BASENAME}_evidence.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG} "sycl"

proc_count="4"
ppn="2"
expected_nodes=$((proc_count / ppn))
prov="$(get_default_prov)"

bench_options="-w 0 -i 1 -l allreduce -y 1024 $(get_default_bench_dtype)"

# 1. Functional: the communicator comes up over OCCP and the collective passes.
cmd="CCL_KVS_MODE=occp CCL_ATL_TRANSPORT=ofi FI_PROVIDER=${prov}"
cmd+=" mpiexec -l -n ${proc_count} -ppn ${ppn} ${SCRIPT_DIR}/benchmark"
cmd+=" ${bench_options} -b sycl > ${TEST_LOG} 2>&1"
run_cmd "${cmd}"
check_log ${TEST_LOG}

# 2. Evidence: prove OCCP carried the bootstrap and that it spanned the nodes.
#    Kept as a separate run so the info/debug output does not reach check_log,
#    whose error patterns match ordinary log lines.
cmd="CCL_KVS_MODE=occp CCL_ATL_TRANSPORT=ofi FI_PROVIDER=${prov}"
cmd+=" CCL_LOG_LEVEL=info CCL_OCCP_LOG_LEVEL=info"
cmd+=" mpiexec -l -n ${proc_count} -ppn ${ppn} ${SCRIPT_DIR}/benchmark"
cmd+=" ${bench_options} -b sycl > ${EVIDENCE_LOG} 2>&1"
run_cmd "${cmd}"

for pattern in "CCL_KVS_MODE: occp" "occp server listens on occp:"
do
    if ! grep -q "${pattern}" ${EVIDENCE_LOG}
    then
        echo "Error: OCCP did not carry the bootstrap, '${pattern}' not found in ${EVIDENCE_LOG}"
        echo "Fail"
        exit 1
    fi
done

# The server picks one forwarder per node and logs it, but it does so again for
# every OCCP session (rank data, ofi config, per communicator), so the number of
# those lines tracks sessions rather than nodes and says nothing about spread.
# Count the distinct addresses instead.
#
# The OCCP logger leaves its "{}" placeholders unsubstituted and appends the
# arguments, so the address is matched as an IPv4 literal rather than by
# position: "added new forwarder: rank:{} ({}) 0 10.0.0.1:40595". On a cluster
# bootstrapping over IPv6 this finds nothing and the check fails, which is the
# safe direction - it will not pass silently.
nodes_seen=$(grep "added new forwarder" ${EVIDENCE_LOG} \
             | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | sort -u | wc -l)
if [ "${nodes_seen}" -lt "${expected_nodes}" ]
then
    echo "Error: OCCP reported ${nodes_seen} distinct node address(es), expected ${expected_nodes}."
    echo "       Either the ranks did not spread over the allocation, or the"
    echo "       forwarder log line changed shape. See ${EVIDENCE_LOG}"
    echo "Fail"
    exit 1
fi

rm ${EVIDENCE_LOG}
rm ${TEST_LOG}
echo "Pass"
