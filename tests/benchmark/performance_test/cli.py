"""Argument parsing and orchestration for run_occl_pt."""

import argparse
import shlex
import sys
from pathlib import Path

from config import COLL_TO_TARGET, MACHINES, METRICS, REMOTE_ENV_USE_DEFAULT
from csv_store import (
    append_command,
    append_run,
    commands_csv_path,
    has_data_rows,
    load_commands,
    metric_csv_path,
    sidecar_dir,
    validate_run_label,
)
from parsing import parse_metric_rows
from runner import apply_machine_preset, build_run_occl_test_argv, run_via_bash
from sizes import build_size_invocations
from xlsx_report import write_xlsx

# (flag spelling for messages, argparse dest) for every option that only means
# something when a benchmark actually runs. --xlsx_from_csv runs none, so
# passing any of these alongside it would silently do nothing -- and hand back a
# rebuilt copy of the *previous* sweep as though it were the one just requested.
SWEEP_ONLY_FLAGS = (
    ("--coll", "coll"),
    ("-n/--nranks", "nranks"),
    ("-g/--ngpus", "ngpus"),
    ("--size", "size"),
    ("--size_list", "size_list"),
    ("--size_range", "size_range"),
    ("--step_factor", "step_factor"),
    ("--dtype", "dtype"),
    ("--op", "op"),
    ("--iters", "iters"),
    ("--warmup", "warmup"),
    ("--root", "root"),
    ("--inplace", "inplace"),
    ("--correctness", "correctness"),
    ("-d/--debug", "debug"),
    ("--machine", "machine"),
    ("--remote", "remote"),
    ("--srun", "srun"),
    ("--remote_env", "remote_env"),
    ("-e/--env", "env"),
    ("--run_label", "run_label"),
    ("--place", "place"),
    ("--dry_run", "dry_run"),
)


def handle_args():
    """Parse the run_occl_pt command line into an argparse Namespace."""
    parser = argparse.ArgumentParser(
        description="Run a oneCCL v2 perf benchmark via run_occl_test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coll", choices=sorted(COLL_TO_TARGET),
                        help="Collective to benchmark. Required except with "
                             "--xlsx_from_csv, which runs nothing.")
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
                        help="Machine preset: expands to --remote/--srun for a "
                             "known box+GPU. Explicit --remote/--srun override "
                             "it. (A preset may also carry -e vars, which the "
                             "explicit ones then override; none currently do.)")
    parser.add_argument("--remote",
                        help="Run on HOST via ssh (passed to run_occl_test).")
    parser.add_argument("--srun",
                        help="srun ARGS string (passed to run_occl_test), "
                             "e.g. '-p bmg-b60 -N 1 --time=00:10:00'.")
    parser.add_argument("--remote_env", nargs="?",
                        const=REMOTE_ENV_USE_DEFAULT, metavar="PATH",
                        help="Requires --remote: source this script on the "
                             "target before the run (run_occl_test "
                             "--remote-env). Bare flag uses "
                             "scripts/remote_env.sh; a relative PATH resolves "
                             "against the repo root. Needed when ssh'ing "
                             "straight to a compute node, which inherits no "
                             "toolchain (srun does, via --export=ALL).")
    parser.add_argument("-e", "--env", action="append", default=[],
                        metavar="VAR=val",
                        help="Env var for the run, forwarded as run_occl_test -e "
                             "(repeatable). On BMG, add -e NEOReadDebugKeys=1 "
                             "-e RenderCompressedBuffersEnabled=0 only if a "
                             "sweep reports nonzero #wrong: they cost device "
                             "memory, so CI omits them by default.")

    parser.add_argument("--run_label", default="",
                        help="Name for this run's column (default: 'runN').")
    # These three name the same results path. The pivot CSVs live in a sidecar
    # dir '<name>_data/' beside it, with one trailing '.xlsx' stripped, so a
    # --csv_only sweep and a later --xlsx_from_csv rebuild share one set of CSVs.
    # They differ only in which half of the work is done.
    out_mode = parser.add_mutually_exclusive_group()
    out_mode.add_argument("--xlsx", metavar="PATH",
                          help="Capture results and export them to this XLSX "
                               "file, one tab and chart per metric: algorithm/"
                               "bus bandwidth and latency (requires openpyxl). "
                               "The per-metric pivot CSVs are written beside it "
                               "in a <name>_data/ sidecar dir as algbw.csv etc.; "
                               "reusing the same path appends a run column.")
    out_mode.add_argument("--csv_only", metavar="PATH",
                          help="Capture results to the pivot CSVs only, skipping "
                               "the workbook: <name>_data/{algbw,busbw,latency}"
                               ".csv plus commands.csv. For hosts without "
                               "openpyxl; the layout matches --xlsx, so the same "
                               "path can be handed to --xlsx_from_csv elsewhere. "
                               "Avoid a dot in the name: only '.xlsx' is "
                               "stripped, so 'sweep.v2' and 'sweep.v3' would "
                               "share one sidecar dir.")
    out_mode.add_argument("--xlsx_from_csv", metavar="PATH",
                          help="Build PATH's workbook from the pivot CSVs "
                               "already in its sidecar dir, then exit ('.xlsx' "
                               "is appended if PATH lacks it). Runs no "
                               "benchmark, and rejects the flags that describe "
                               "one. Use after a --csv_only sweep, on a host "
                               "that has openpyxl.")
    parser.add_argument("--place", default="oop", choices=("oop", "ip"),
                        help="Which measurement to capture: out-of-place (oop) "
                             "or in-place (ip). In-place needs --inplace.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the run_occl_test command(s) and exit.")
    args = parser.parse_args()

    # Reject a results path that cannot name a workbook before anything runs: an
    # empty value (an unset variable in a wrapper script) would otherwise read as
    # "flag not given" and throw a whole sweep away at exit 0, and a directory
    # would raise out of sidecar_dir() only after the sweep finished.
    for flag, value in (("--xlsx", args.xlsx), ("--csv_only", args.csv_only),
                        ("--xlsx_from_csv", args.xlsx_from_csv)):
        if value is None:
            continue
        if not value.strip():
            parser.error(f"{flag} needs a path (got an empty value)")
        if not Path(value).name:
            parser.error(f"{flag} needs a path to a file, not a directory "
                         f"({value!r} names none); the pivot CSVs go in a "
                         f"sibling <name>_data/ dir")
    if args.remote_env is not None and not args.remote_env.strip():
        parser.error("--remote_env needs a PATH, or pass it bare to use "
                     "scripts/remote_env.sh")

    if args.xlsx_from_csv:
        # Fail loudly rather than rebuild the previous sweep and call it this
        # one. Compared against each flag's default so store_true and append
        # flags are covered too; the blind spot is passing a flag its own
        # default value, which is a no-op anyway.
        given = [flag for flag, dest in SWEEP_ONLY_FLAGS
                 if getattr(args, dest) != parser.get_default(dest)]
        if given:
            parser.error(f"--xlsx_from_csv rebuilds a workbook from existing "
                         f"pivot CSVs and runs no benchmark, so it cannot be "
                         f"combined with {', '.join(given)}")
    else:
        if not args.coll:
            parser.error("--coll is required (except with --xlsx_from_csv)")
        # run_occl_test rejects --remote-env without --remote, but only once the
        # sweep is already dispatching -- once per size invocation, each failure
        # reported against a size range rather than the real cause.
        if args.remote_env and not args.remote and not (
                args.machine and MACHINES[args.machine].get("remote")):
            parser.error("--remote_env requires --remote HOST (or a --machine "
                         "preset that supplies one)")
    return args


def build_xlsx_from_csvs(results_path):
    """Build the workbook from the pivot CSVs an earlier --csv_only/--xlsx sweep
    left in the sidecar dir, running no benchmark. Second half of the split-host
    flow: sweep with --csv_only where openpyxl is missing, then --xlsx_from_csv
    where it isn't. The workbook is written to `results_path` with '.xlsx'
    appended when absent, so the base a --csv_only sweep used can be handed
    straight over."""
    metric_csvs = [(metric, metric_csv_path(results_path, metric.name))
                   for metric in METRICS]
    # Existence is not enough: a CSV that is empty or header-only (a truncated
    # transfer of the sidecar dir, or a metric that scraped nothing) yields no
    # tab, so gating on exists() alone would report success having written a
    # workbook with a metric silently missing -- or no workbook at all.
    unusable = [str(path) for _, path in metric_csvs if not has_data_rows(path)]
    if unusable:
        sys.exit(f"Error: --xlsx_from_csv found no usable pivot CSVs at "
                 f"{', '.join(unusable)} (missing, empty or header-only); run a "
                 f"--csv_only or --xlsx sweep first.")
    xlsx_path = Path(results_path)
    if xlsx_path.suffix.lower() != ".xlsx":
        xlsx_path = xlsx_path.with_name(f"{xlsx_path.name}.xlsx")
    print(f"[run_occl_pt] rebuilding from {sidecar_dir(results_path)}")
    commands = load_commands(commands_csv_path(results_path))
    if not write_xlsx(xlsx_path, metric_csvs, commands):
        sys.exit(f"Error: no workbook written; the pivot CSVs in "
                 f"{sidecar_dir(results_path)} hold no chartable rows.")


def main():
    """Parse args, run the perf binary once per size invocation, and (with
    --xlsx/--csv_only) accumulate the scraped bandwidth rows into pivot CSVs,
    and (with --xlsx) an XLSX."""
    args = handle_args()
    if args.xlsx_from_csv:
        build_xlsx_from_csvs(args.xlsx_from_csv)
        return
    if args.place == "ip" and not args.inplace:
        sys.exit("Error: --place ip needs --inplace (the binary only emits "
                 "in-place results when run with -p 1).")
    # --xlsx and --csv_only name the same results path; only --xlsx also builds
    # the workbook on top of the pivot CSVs.
    results_path = args.xlsx or args.csv_only
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
        if results_path:
            for size, values in parse_metric_rows(lines, args.place).items():
                for name, sizes in run_metrics.items():
                    if values.get(name):
                        sizes[size] = values[name]

    metric_csvs = []  # (Metric, csv_path) for the XLSX
    if results_path and any(run_metrics.values()):
        col_name = None
        for metric in METRICS:
            path = metric_csv_path(results_path, metric.name)
            # Same column name across metrics (the files evolve in lockstep).
            col_name = append_run(path, args.run_label, run_metrics[metric.name])
            metric_csvs.append((metric, path))
            print(f"[run_occl_pt] appended column '{col_name}' to {path}")
        # Persist this run's own invocation, keyed by the column name just
        # assigned, so the XLSX can show a command for every accumulated run.
        cmd = "python3 " + shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])
        append_command(commands_csv_path(results_path), col_name, cmd)

    if results_path and not args.dry_run and not metric_csvs:
        # Runs that exited 0 but whose rows did not match the expected table
        # format (e.g. the benchmark's columns changed). Report it on both
        # capture paths: under --csv_only there is no later XLSX step to notice,
        # so the sweep would otherwise look successful and leave nothing behind.
        print("[run_occl_pt] no results parsed; nothing captured.",
              file=sys.stderr)
        exit_code = exit_code or 1
    elif args.xlsx and not args.dry_run:
        write_xlsx(args.xlsx, metric_csvs,
                   load_commands(commands_csv_path(args.xlsx)))

    sys.exit(exit_code)
