"""Render the parity report the test suite writes with ``--parity-report``.

Each line of the report is one tensor comparison: how far the port sat from the
reference, as a ratio of the calibrated bound (1.0 is the tolerance itself,
above it the assertion failed). A run's number is therefore comparable across
sizes, devices and build flags, which is what makes it possible to say *how
much* a faster build misses by instead of only that it does.

    ./cpp/build/tests/muscriptor_tests --parity-report parity.jsonl
    uv run msl-parity-report parity.jsonl                 # worst tensors first
    uv run msl-parity-report fast.jsonl parity.jsonl      # side by side, one column per file
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> dict[tuple[str, str], dict]:
    """@return Lines keyed by (config, tensor); a repeated key keeps its worst ratio."""
    rows: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["config"], row["tensor"])
            if key not in rows or _worst(row) > _worst(rows[key]):
                rows[key] = row
    return rows


def _worst(row: dict) -> float:
    if row.get("nonfinite_mismatch"):
        return float("inf")
    return max(row["max_abs_ratio"], row["cosine_ratio"])


def _fmt(ratio: float) -> str:
    if ratio == float("inf"):
        return "non-finite"
    return f"{ratio:8.3f}" + ("  FAIL" if ratio > 1.0 else "")


def _summary(rows: dict[tuple[str, str], dict]) -> dict[str, tuple[int, int, float]]:
    """@return Per config: (tensors compared, tensors over budget, worst ratio)."""
    out: dict[str, tuple[int, int, float]] = {}
    per_config: dict[str, list[float]] = defaultdict(list)
    for (config, _), row in rows.items():
        per_config[config].append(_worst(row))
    for config, ratios in sorted(per_config.items()):
        out[config] = (len(ratios), sum(r > 1.0 for r in ratios), max(ratios))
    return out


def render(paths: list[Path], limit: int) -> str:
    reports = [(p.stem, _load(p)) for p in paths]
    labels = [label for label, _ in reports]
    width = max(len(label) for label in labels)
    lines: list[str] = []

    lines.append("summary  (ratio of the calibrated bound; 1.0 is the tolerance)")
    lines.append(f"  {'report':<{width}}  {'configuration':<18} {'tensors':>8} {'over':>6} {'worst':>10}")
    for label, rows in reports:
        for config, (count, over, worst) in _summary(rows).items():
            lines.append(
                f"  {label:<{width}}  {config:<18} {count:>8} {over:>6} {_fmt(worst):>10}"
            )

    # One table per configuration, tensors as rows, reports as columns, ordered
    # by the worst ratio across the columns so a regression floats to the top.
    configs = sorted({config for _, rows in reports for config, _ in rows})
    for config in configs:
        tensors = sorted(
            {tensor for _, rows in reports for c, tensor in rows if c == config},
            key=lambda t: -max(
                _worst(rows[(config, t)]) for _, rows in reports if (config, t) in rows
            ),
        )
        lines.append("")
        lines.append(f"{config}  (max of max_abs and cosine ratios; worst {limit} of {len(tensors)})")
        header = f"  {'tensor':<28}" + "".join(f" {label:>16}" for label in labels)
        lines.append(header)
        for tensor in tensors[:limit]:
            cells = []
            for _, rows in reports:
                row = rows.get((config, tensor))
                cells.append(f" {_fmt(_worst(row)) if row else '-':>16}")
            lines.append(f"  {tensor:<28}" + "".join(cells))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="JSON-lines files from --parity-report")
    parser.add_argument("--worst", type=int, default=20, help="tensors to list per configuration")
    args = parser.parse_args()
    print(render(args.reports, args.worst))
