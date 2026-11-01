"""Accumulator CSV sidecars for run_occl_pt.

The pivot CSVs (one per metric) and the commands CSV are derived from the
--xlsx path and live in a per-XLSX subdir `<base>_data/`, so re-running with
the same --xlsx appends onto the same tables. openpyxl is not needed here."""

import csv
import sys
from pathlib import Path

# A leading one of these makes a spreadsheet treat a cell as a formula, so a
# run label starting with one would be executed rather than shown when the CSV
# or XLSX is opened (CSV/XLSX injection). Rejected outright -- run labels are
# column names, they have no reason to start with these.
FORMULA_LEAD_CHARS = "=+-@"


def validate_run_label(label):
    """Exit with an error if `label` would be read as a formula by a
    spreadsheet. Called before running anything so a bad label fails fast
    rather than after a long sweep."""
    stripped = label.strip()
    if stripped and stripped[0] in FORMULA_LEAD_CHARS:
        sys.exit(f"Error: --run_label '{label}' starts with '{stripped[0]}'; "
                 f"spreadsheets read such cells as formulas. Pick a label that "
                 f"starts with a letter or digit.")


def sidecar_dir(xlsx_path):
    """Directory holding this XLSX's sidecar CSVs: '<dir>/<base>_data'."""
    path = Path(xlsx_path)
    return path.with_name(f"{path.stem}_data")


def metric_csv_path(xlsx_path, metric_name):
    """Per-metric pivot CSV path, e.g. bench.xlsx + 'algbw' ->
    bench_data/algbw.csv."""
    return sidecar_dir(xlsx_path) / f"{metric_name}.csv"


def commands_csv_path(xlsx_path):
    """Sidecar commands CSV path, e.g. bench.xlsx -> bench_data/commands.csv.
    Maps each run's column name to the invocation that produced it, and
    accumulates across runs so every run's command survives an XLSX rebuild."""
    return sidecar_dir(xlsx_path) / "commands.csv"


def _is_populated(path):
    """True if `path` exists with content, i.e. holds a header to extend."""
    return path.exists() and path.stat().st_size > 0


def append_command(path, col_name, command):
    """Append (col_name, command) to the sidecar commands CSV, creating it with
    a header on first use. Keyed by the same column name append_run assigns, so
    the XLSX can pair each run's chart series with its command."""
    path = Path(path)
    populated = _is_populated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not populated:
            writer.writerow(["run", "command"])
        writer.writerow([col_name, command])


def load_commands(path):
    """Read the sidecar commands CSV into an ordered list of (col_name,
    command). Returns [] if the file is missing/empty."""
    path = Path(path)
    if not _is_populated(path):
        return []
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return [(row[0], row[1]) for row in rows[1:] if len(row) >= 2]


def append_run(path, label, size_to_val):
    """Append this run to the per-metric pivot CSV at `path` as a new column.

    The file is shaped `size | run1 | run2 | ...`. On first use it is created
    with a 'size' header; each later run adds one more run column, unioning
    sizes (blank cells where a run has no value for a size). This is what makes
    re-running with the same --xlsx accumulate onto one table (and one chart).
    The new column is named by `label`, or 'runN' (N = number of runs so far)
    when no label is given. Returns the new column's name."""
    path = Path(path)
    header = ["size"]
    rows = {}  # size -> list of existing per-run values (strings)
    if _is_populated(path):
        with path.open(newline="") as f:
            data = [row for row in csv.reader(f) if row]
        if data:
            header = data[0]
            rows = {int(row[0]): row[1:] for row in data[1:]}

    n_run_cols = len(header) - 1
    col_name = label or f"run{n_run_cols + 1}"
    if col_name in header[1:]:
        sys.exit(f"Error: run column '{col_name}' already exists in {path}; "
                 f"pass a distinct --run_label.")
    header.append(col_name)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for size in sorted(rows.keys() | size_to_val.keys()):
            prev = rows.get(size, [])
            prev += [""] * (n_run_cols - len(prev))  # pad ragged old rows
            writer.writerow([size, *prev, size_to_val.get(size, "")])
    return col_name
