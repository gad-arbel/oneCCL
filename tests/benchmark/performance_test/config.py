"""Static config tables and compiled regexes for run_occl_pt."""

import re
from typing import NamedTuple

# --coll name -> run_occl_test target. Names in dev_test.sh's static map
# are passed bare; the rest go through as a build-relative path (which
# run_occl_test resolves via its "*/*" case).
COLL_TO_TARGET = {
    "broadcast": "broadcast_perf",
    "allreduce": "allreduce_perf",
    "allgather": "allgather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "sendrecv": "sendrecv_perf",
    "alltoall": "tests/benchmark/alltoall_perf",
    "scatter": "tests/benchmark/scatter_perf",
}

# Machine presets: one --machine NAME expands to the remote/srun/env boilerplate
# for a given box + GPU family, so the everyday invocation stays short (and the
# command recorded in the XLSX is the reproducible short form). Explicit
# --remote/--srun override the preset; explicit -e vars are appended after the
# preset's. To add a machine, add one entry. BMG/Arc B-series needs the two NEO
# driver keys (see CLAUDE.md): NEOReadDebugKeys=1 gates the driver's debug keys,
# RenderCompressedBuffersEnabled=0 disables buggy render-buffer compression.
MACHINES = {
    "occl-gta-bmg": {
        "remote": "occl-gta",
        "srun": "-p bmg-b60 -N 1 --time=00:10:00",
        "env": ["NEOReadDebugKeys=1", "RenderCompressedBuffersEnabled=0"],
    },
}

# Matches the binary's benchmark data rows. The C++ prints, per size:
#   size count type redop root   time algbw busbw wrong   [time algbw busbw wrong]
# (the second quad is the in-place run, present when inplace is on).
# type/redop are strings (e.g. fp32/sum), the rest numeric. Only the values we
# chart are captured, named after their metric: the binary's 'time' column is
# captured as 'latency', and the in-place quad repeats the names with an 'ip_'
# prefix (so parsing.py can pick a quad by prefixing the metric name).
ROW_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+\d+\s+\w+\s+\w+\s+-?\d+"  # size count type redop root
    r"\s+(?P<latency>[\d.]+)\s+(?P<algbw>[\d.]+)\s+(?P<busbw>[\d.]+)\s+\S+"
    r"(?:\s+(?P<ip_latency>[\d.]+)\s+(?P<ip_algbw>[\d.]+)"
    r"\s+(?P<ip_busbw>[\d.]+)\s+\S+)?\s*$"
)


class Metric(NamedTuple):
    """One charted metric. `name` doubles as its ROW_RE group name, its pivot
    CSV stem and its key in the scraped per-size dicts."""

    name: str
    tab_title: str
    y_label: str


# Metrics captured from each data row. Each becomes its own pivot CSV
# (size x run) and its own tab + chart in the XLSX. To add a metric, add one
# entry here plus its ROW_RE group -- capture, CSV split, tabs and charting are
# all metric-driven.
METRICS = (
    Metric("algbw", "Algorithm Bandwidth", "Algorithm bandwidth (GB/s)"),
    Metric("busbw", "Bus Bandwidth", "Bus bandwidth (GB/s)"),
    # Per-iteration collective latency. Lower is better, unlike the bandwidths.
    Metric("latency", "Latency", "Latency (us)"),
)

# oneCCL log lines look like "2026:07:29-15:57:45:42845 |CCL_WARN| ...".
# They interleave with the benchmark's own table and just add noise, so they
# are hidden from the live echo (but still captured). ERROR is kept -- if a run
# breaks, the reason should stay on screen.
CCL_LOG_RE = re.compile(r"\|CCL_(WARN|INFO|DEBUG|TRACE)\|")
