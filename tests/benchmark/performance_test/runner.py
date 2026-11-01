"""Compose and execute the run_occl_test invocation for run_occl_pt.

run_occl_test is a bash function in scripts/dev_test.sh; we source it and call
the function, streaming output live while capturing it for parsing."""

import shlex
import subprocess
import sys
from pathlib import Path

from config import CCL_LOG_RE, COLL_TO_TARGET, MACHINES


def build_run_occl_test_argv(args, min_bytes, max_bytes, step_factor):
    """Compose the run_occl_test argument vector for one invocation."""
    target = COLL_TO_TARGET[args.coll]
    flavor = "-d" if args.debug else "-r"

    argv = ["run_occl_test", flavor]
    if args.remote:
        argv += ["--remote", args.remote]
    if args.srun:
        argv += ["--srun", args.srun]
    for kv in args.env:
        argv += ["-e", kv]
    argv += ["-n", str(args.nranks), target, "--"]

    # benchmark binary flags (see tests/benchmark/common.cpp getopt table)
    bench = [
        "-b", str(min_bytes),
        "-e", str(max_bytes),
        "-f", str(step_factor),
        "-n", str(args.iters),
        "-w", str(args.warmup),
        "-d", args.dtype,
        "-o", args.op,
        "-c", "1" if args.correctness else "0",
        "-p", "1" if args.inplace else "0",
    ]
    if args.ngpus is not None:
        bench += ["-g", str(args.ngpus)]
    if args.root is not None:
        bench += ["-r", str(args.root)]
    return argv + bench


def run_via_bash(argv):
    """run_occl_test is a bash function in scripts/dev_test.sh, so it can't
    be exec'd directly. Source the script, then call the function. Streams
    output live (minus oneCCL's noisy log lines) while capturing every line
    for CSV parsing.

    Output is streamed as raw bytes rather than decoded text. Under --remote,
    run_occl_test runs `ssh -tt`, which forces a remote pty (its output uses
    CRLF line endings) and switches the local terminal to raw mode (bare LF no
    longer returns the cursor to column 0). Piping that through Python's text
    mode strips the CR, so the re-emitted bare LF walks each row diagonally
    across the screen. Passing the bytes through untouched keeps the CRLF the
    terminal needs; the decode for parsing is a throwaway copy."""
    repo_root = Path(__file__).resolve().parents[3]
    dev_test = repo_root / "scripts" / "dev_test.sh"
    bash_cmd = f"source {shlex.quote(str(dev_test))} && {shlex.join(argv)}"

    lines = []
    with subprocess.Popen(
        ["bash", "-c", bash_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # bytes, no newline xlate
    ) as proc:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace")
            lines.append(line)  # capture all lines (decoded) for parsing
            if CCL_LOG_RE.search(line):
                continue  # hide noisy WARN/INFO/DEBUG/TRACE from the live echo
            sys.stdout.buffer.write(raw)  # pass bytes through so CRLF survives
            sys.stdout.flush()
    return proc.returncode, lines


def apply_machine_preset(args):
    """Fill --remote/--srun/-e from the named --machine preset. Explicit flags
    win: --remote/--srun set on the command line are left untouched; preset env
    vars are prepended so any explicit -e still applies (and can override)."""
    if not args.machine:
        return
    preset = MACHINES[args.machine]
    if not args.remote:
        args.remote = preset.get("remote")
    if not args.srun:
        args.srun = preset.get("srun")
    args.env = list(preset.get("env", [])) + args.env
