"""Message-size parsing and per-invocation size expansion for run_occl_pt."""

import sys

UNITS = {"G": 1024 ** 3, "M": 1024 ** 2, "K": 1024, "B": 1}


def parse_size(size):
    """Parse '128M' / '1K' / '4096' (B/K/M/G, case-insensitive) to an int
    byte count."""
    try:
        unit = size[-1].upper()
        if not unit.isalpha():
            return int(size)
        number = float(size[:-1])
    except (IndexError, ValueError):
        sys.exit(f"Error: could not parse size '{size}'.")
    if unit not in UNITS:
        print(f"[run_occl_pt] unsupported unit '{unit}' in '{size}', "
              f"treating as bytes.", file=sys.stderr)
        return int(number)
    return int(number * UNITS[unit])


def build_size_invocations(size, size_list, size_range, step_factor):
    """Return a list of (min_bytes, max_bytes, step_factor) tuples, one per
    run_occl_test invocation. size_list runs the binary once per size;
    size / size_range run it once and let the binary sweep."""
    if size_list:
        sizes = [parse_size(s.strip()) for s in size_list.split(",") if s.strip()]
        return [(s, s, 1) for s in sizes]
    if size_range:
        low, high = (parse_size(s) for s in size_range)
        if low > high:
            sys.exit(f"Error: size_range min ({low}) > max ({high}).")
        if step_factor < 2:
            sys.exit(f"Error: --step_factor must be >= 2 for --size_range "
                     f"(got {step_factor}); the binary's sweep would not "
                     f"advance.")
        return [(low, high, step_factor)]
    parsed = parse_size(size)
    return [(parsed, parsed, 1)]
