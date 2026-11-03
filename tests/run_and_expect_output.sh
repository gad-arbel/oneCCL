#!/usr/bin/env bash

set -uo pipefail

if (( $# < 2 )); then
    echo "Usage: run_and_expect_output.sh <expected-output> <command> [args...]" >&2
    exit 2
fi

readonly expected_output=$1
shift

output_file=$(mktemp) || exit 1
readonly output_file
trap 'rm -f "${output_file}"' EXIT

"$@" 2>&1 | tee "${output_file}"
readonly command_status=${PIPESTATUS[0]}

if (( command_status != 0 )); then
    exit "${command_status}"
fi

if ! grep -Fq -- "${expected_output}" "${output_file}"; then
    echo "Expected output was not found: ${expected_output}" >&2
    exit 1
fi
