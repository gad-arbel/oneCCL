"""Argument parsing and orchestration for run_occl_pt."""

import argparse
import shlex
import sys
from pathlib import Path

from config import COLL_TO_TARGET, MACHINES, METRICS
from csv_store import (
    append_command,
    append_run,
    commands_csv_path,
    load_commands,
    metric_csv_path,
    validate_run_label,
)
from parsing import parse_metric_rows
from runner import apply_machine_preset, build_run_occl_test_argv, run_via_bash
from sizes import build_size_invocations
from xlsx_report import write_xlsx


def handle_args():
    """Parse the run_occl_pt command line into an argparse Namespace."""
    parser = argparse.ArgumentParser(
        description="Run a oneCCL v2 perf benchmark via run_occl_test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coll", required=True,
                        choices=sorted(COLL_TO_TARGET),
                        help="Collective to benchmark.")
    parser.add_argument("-n", "--nranks", type=int, default=2,
                        help="Number of ranks.")
    parser.add_argument("-g", "--ngpus", type=int,
                        help="GPUs per rank (binary -g). Default: binary default.")

    size = parser.add_mutually_exclusive_group()
    size.add_argument("--size", default="1M",
                      help="Single message size (B/K/M/G suffix).")
    size.add_argument("--size_list",
                      help="Comma-separated sizes, one run each, e.g. 4M,8M,16M.")
    size.add_argument("--size_range", nargs=2, metavar=("MIN", "MAX"),
                      help="Sweep MIN..MAX (B/K/M/G), stepping by --step_factor.")
    parser.add_argument("--step_factor", type=int, default=2,
                        help="Multiplier per step for --size_range.")

    parser.add_argument("--dtype", default="fp32",
                        help="Data type: int8/uint8/int32/uint32/int64/uint64/"
                             "fp16/fp32/fp64/bf16.")
    parser.add_argument("--op", default="sum",
                        choices=("sum", "prod", "max", "min"),
                        help="Reduction op (for reducing collectives).")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations.")
    parser.add_argument("--root", type=int,
                        help="Root rank (broadcast/scatter).")
    parser.add_argument("--inplace", action="store_true",
                        help="Also run in-place.")
    parser.add_argument("--correctness", action="store_true",
                        help="Enable correctness check.")

    parser.add_argument("-d", "--debug", action="store_true",
                        help="Use build_debug/ binaries (default: build_release/).")
    parser.add_argument("--machine", choices=sorted(MACHINES),
                        help="Machine preset: expands to --remote/--srun/-e for "
                             "a known box+GPU. Explicit --remote/--srun override "
                             "it; explicit -e vars are appended.")
    parser.add_argument("--remote",
                        help="Run on HOST via ssh (passed to run_occl_test).")
    parser.add_argument("--srun",
                        help="srun ARGS string (passed to run_occl_test), "
                             "e.g. '-p bmg-b60 -N 1 --time=00:10:00'.")
    parser.add_argument("-e", "--env", action="append", default=[],
                        metavar="VAR=val",
                        help="Env var for the run, forwarded as run_occl_test -e "
                             "(repeatable). BMG needs -e NEOReadDebugKeys=1 "
                             "-e RenderCompressedBuffersEnabled=0.")

    parser.add_argument("--run_label", default="",
                        help="Name for this run's column (default: 'runN').")
    parser.add_argument("--xlsx",
                        help="Export results to this XLSX file, one tab and "
                             "chart per metric: algorithm/bus bandwidth and "
                             "latency (requires openpyxl). The per-metric pivot "
                             "CSVs are written in a <base>_data/ sidecar dir as "
                             "algbw.csv etc.; reusing the same --xlsx appends a "
                             "run column.")
    parser.add_argument("--place", default="oop", choices=("oop", "ip"),
                        help="Which measurement to capture: out-of-place (oop) "
                             "or in-place (ip). In-place needs --inplace.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the run_occl_test command(s) and exit.")
    return parser.parse_args()


def main():
    """Parse args, run the perf binary once per size invocation, and (with
    --xlsx) accumulate the scraped bandwidth rows into pivot CSVs and an XLSX."""
    args = handle_args()
    if args.place == "ip" and not args.inplace:
        sys.exit("Error: --place ip needs --inplace (the binary only emits "
                 "in-place results when run with -p 1).")
    validate_run_label(args.run_label)
    apply_machine_preset(args)
    invocations = build_size_invocations(
        args.size, args.size_list, args.size_range, args.step_factor
    )

    # Accumulate across all invocations of this run: per-metric {size: value}.
    # (--size_list issues one binary run per size; they form one run column.)
    run_metrics = {metric.name: {} for metric in METRICS}
    exit_code = 0
    for min_bytes, max_bytes, step_factor in invocations:
        argv = build_run_occl_test_argv(args, min_bytes, max_bytes, step_factor)
        if args.dry_run:
            # Quote as run_via_bash does, so the printed line is copy-pasteable
            # (--srun takes one multi-word argument).
            print(shlex.join(argv))
            continue
        returncode, lines = run_via_bash(argv)
        if returncode != 0:
            print(f"[run_occl_pt] run failed (exit {returncode}) for size range "
                  f"{min_bytes}..{max_bytes}; not recording its output",
                  file=sys.stderr)
            exit_code = returncode
            continue  # don't pollute the pivot table with a failed run's rows
        if args.xlsx:
            for size, values in parse_metric_rows(lines, args.place).items():
                for name, sizes in run_metrics.items():
                    if values.get(name):
                        sizes[size] = values[name]

    metric_csvs = []  # (Metric, csv_path) for the XLSX
    if args.xlsx and any(run_metrics.values()):
        col_name = None
        for metric in METRICS:
            path = metric_csv_path(args.xlsx, metric.name)
            # Same column name across metrics (the files evolve in lockstep).
            col_name = append_run(path, args.run_label, run_metrics[metric.name])
            metric_csvs.append((metric, path))
            print(f"[run_occl_pt] appended column '{col_name}' to {path}")
        # Persist this run's own invocation, keyed by the column name just
        # assigned, so the XLSX can show a command for every accumulated run.
        cmd = "python3 " + shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
        append_command(commands_csv_path(args.xlsx), col_name, cmd)

    if args.xlsx and not args.dry_run:
        if metric_csvs:
            commands = load_commands(commands_csv_path(args.xlsx))
            write_xlsx(args.xlsx, metric_csvs, commands)
        else:
            print("[run_occl_pt] no results parsed; skipping XLSX.",
                  file=sys.stderr)

    sys.exit(exit_code)
