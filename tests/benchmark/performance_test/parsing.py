"""Parse the perf binary's benchmark data rows into per-size metric values."""

from config import METRICS, ROW_RE


def parse_metric_rows(lines, place):
    """Extract benchmark data rows into {size_bytes: {metric_name: value}} for
    the chosen place ('oop' out-of-place or 'ip' in-place). Values are the raw
    strings from the output, or None when the place wasn't run (the in-place
    quad is absent, so its groups never match)."""
    prefix = "ip_" if place == "ip" else ""
    rows = {}
    for line in lines:
        match = ROW_RE.match(line)
        if match:
            rows[int(match["size"])] = {
                metric.name: match[prefix + metric.name] for metric in METRICS
            }
    return rows
