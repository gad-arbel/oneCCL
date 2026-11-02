#!/bin/bash

# remove duplicate "/"
simplify_path() {
    echo $1 | sed "s/\/\+/\//g"
}


INSTALL_DIRECTORY=$( cd "$1" && pwd -P )
SOURCE_VARS="source $( simplify_path ${INSTALL_DIRECTORY}/env/vars.sh ) "
SOURCE_SETVARS="source $( simplify_path  ${INSTALL_DIRECTORY}/env/setvars.sh ) "

BUNDLED_MPI_YES="--ccl-bundled-mpi=yes"
BUNDLED_MPI_NO="--ccl-bundled-mpi=no"
BUNDLED_MPI_INVALID="--ccl-bundled-mpi=yas"

BUNDLED_MPI_YES_SINGLE="-ccl-bundled-mpi=yes"
BUNDLED_MPI_NO_SINGLE="-ccl-bundled-mpi=no"
BUNDLED_MPI_INVALID_SINGLE="-ccl-bundled-mpi=yas"

# oneCCL no longer bundles the Intel(R) MPI Library: sourcing must keep the external mpirun
# and must not add any path under opt/mpi
CHECK_MPI="which mpirun && env"
MPI_DEFAULT=$(which mpirun)
MPI_BUNDLED_ROOT="$( simplify_path ${INSTALL_DIRECTORY}/opt/mpi )"

if [[ -z "${MPI_DEFAULT}" ]] ; then
    echo "mpirun not found, put an external MPI on PATH before running this test"
    exit 1
fi

if [[ -d "${MPI_BUNDLED_ROOT}" ]] ; then
    echo "${MPI_BUNDLED_ROOT} exists, test a clean install directory"
    exit 1
fi

ERROR_MESSAGE_SETVARS_FLAG="ccl_bundled_mpi is only supported by direct call vars.sh, ignoring"
WARNING_MESSAGE_NO_BUNDLED_MPI="ccl_bundled_mpi=yes is ignored"

# execute command supplied in arguments in a subshell
# so that env variables in the current shell are not modified
execute_in_subshell_and_check() {
    to_execute=$1
    expected_substring=$2
    unexpected_substring=$3

    command="${to_execute} && ${CHECK_MPI}"
    output=$( bash -c "${command}" )
    if [[ ! "${output}" == *"${expected_substring}"* ]] ; then
        echo "expected part NOT found: ${expected_substring} in ${output}"
        exit 1
    fi
    if [[ "${output}" == *"${unexpected_substring}"* ]] ; then
        echo "unexpected part found: ${unexpected_substring} in ${output}"
        exit 1
    fi
}

# default and explicit flags keep the external mpi
execute_in_subshell_and_check "${SOURCE_SETVARS}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}" # setvars
execute_in_subshell_and_check "${SOURCE_SETVARS}" "${MPI_DEFAULT}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # setvars: no warning
execute_in_subshell_and_check "${SOURCE_VARS}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}"	# default
execute_in_subshell_and_check "${SOURCE_VARS}" "${MPI_DEFAULT}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}"	# default: no warning
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}"	# explicit yes
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}" # explicit no
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO}" "${MPI_DEFAULT}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # explicit no: no warning

# execution errors - flag supplied to setvars
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"

# execution errors - invalid flag
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID}" "unrecognized" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID}" "unrecognized" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # invalid: no warning

# repeat tests with single dash

# explicit single-dash flags keep the external mpi (bare sourcing is covered above)
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES_SINGLE}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}"	# explicit yes
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES_SINGLE}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO_SINGLE}" "${MPI_DEFAULT}" "${MPI_BUNDLED_ROOT}" # explicit no
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO_SINGLE}" "${MPI_DEFAULT}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # explicit no: no warning

# execution errors - flag supplied to setvars
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_BUNDLED_ROOT}"

# execution errors - invalid flag
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID_SINGLE}" "unrecognized" "${MPI_BUNDLED_ROOT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID_SINGLE}" "unrecognized" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # invalid: no warning

# fixture: an install tree upgraded in place from a version that bundled Intel MPI
mkdir -p "${MPI_BUNDLED_ROOT}/bin" "${MPI_BUNDLED_ROOT}/lib"
touch "${MPI_BUNDLED_ROOT}/bin/mpirun"
chmod +x "${MPI_BUNDLED_ROOT}/bin/mpirun"

execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES}" "I_MPI_ROOT=${MPI_BUNDLED_ROOT}" "${WARNING_MESSAGE_NO_BUNDLED_MPI}" # bundled mpi tree present: export I_MPI_ROOT, no warning

rm -rf "${MPI_BUNDLED_ROOT}"

echo "Pass"
