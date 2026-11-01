#!/usr/bin/env python3

"""
run_occl_pt - oneCCL performance tool runner.

A thin front-end over the oneCCL v2 perf binaries
(tests/benchmark/*_perf). It does NOT re-implement launching: it shells
out to `run_occl_test` (scripts/dev_test.sh), which already handles
target resolution, multi-rank mpiexec, env.sh sourcing, --remote HOST
(ssh to a head node) and --srun. On top of that this script adds the two
things run_occl_test does not:

  1. Friendly size selection: --size / --size_list / --size_range with
     B/K/M/G suffixes, translated to the binary's -b/-e/-f flags.
  2. CSV/XLSX capture: with --xlsx the benchmark's data rows are scraped into
     per-metric pivot CSVs (size x run) written in a <base>_data/ sidecar dir
     alongside the XLSX (out.xlsx -> out_data/algbw.csv, out_data/busbw.csv,
     out_data/latency.csv), and each metric gets its own XLSX tab and chart.
     Each run adds one column; re-running with the same --xlsx appends a new run
     column automatically, so runs accumulate onto the same table and chart.
     Each run's own invocation is also recorded in out_data/commands.csv; the
     XLSX lists every run's command below the chart and links each run's column
     header to its command cell.

Usage examples
    # single size, 2 ranks on a remote head node
    python3 run_occl_pt.py --coll broadcast -n 2 --remote occl-gta --size 128M

    # size sweep 1K..1G doubling each step, capture to XLSX
    python3 run_occl_pt.py --coll allreduce -n 8 --size_range 1K 1G --xlsx out.xlsx

    # append a second run onto the same table/chart (reuse --xlsx); each run is
    # its own column/line, named by --run_label (or 'runN').
    python3 run_occl_pt.py --coll allreduce -n 8 --size_range 1K 1G \
        --xlsx out.xlsx --run_label baseline
    python3 run_occl_pt.py --coll allreduce -n 16 --size_range 1K 1G \
        --xlsx out.xlsx --run_label scaled

    # explicit list of sizes (one binary run per size)
    python3 run_occl_pt.py --coll broadcast -n 4 --size_list 4M,8M,16M

    # see the run_occl_test command without running it
    python3 run_occl_pt.py --coll broadcast -n 2 --remote occl-gta --size 1M --dry_run
"""

from cli import main

if __name__ == "__main__":
    main()
