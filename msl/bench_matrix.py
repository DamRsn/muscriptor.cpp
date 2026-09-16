"""Benchmark every size on every device, ggml against PyTorch, one process each.

Runs ``muscriptor_bench --transcribe`` and ``msl-bench-torch --transcribe`` for
each (size, device) pair and prints one markdown table of real-time factors.
Every measurement is its own process, with a pause between them, since a run
that directly follows a PyTorch run reads slow. The PyTorch GPU runs use MPS, so
they need Apple Silicon.

    uv run msl-bench-matrix                       # all sizes with weights, cpu + gpu
    uv run msl-bench-matrix --sizes small,medium --devices gpu --sides ggml
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from msl import paths

SIZES = ("small", "medium", "large")
DEVICES = ("cpu", "gpu")
SIDES = ("ggml", "torch")
# What each side calls the GPU.
TORCH_DEVICE = {"cpu": "cpu", "gpu": "mps"}

_SPEED = re.compile(r"^speed\s+([0-9.]+)x real time", re.M)
_ELAPSED = re.compile(r"^transcribe\s+([0-9.]+) s", re.M)
_BACKEND = re.compile(r"^backend\s+(\S+)", re.M)
_NOTES = re.compile(r"^notes\s+(\d+)", re.M)


def _command(side: str, size: str, device: str, bench: Path) -> list[str]:
    if side == "ggml":
        return [str(bench), "--size", size, "--device", device, "--transcribe"]
    return ["uv", "run", "msl-bench-torch", "--model", size, "--device", TORCH_DEVICE[device], "--transcribe"]


def _selection(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    """@return The comma-separated values of ``--name``, or exits naming the unknown ones."""
    chosen = [v for v in value.split(",") if v]
    unknown = [v for v in chosen if v not in allowed]
    if unknown:
        sys.exit(f"unknown --{name}: {','.join(unknown)}; choose from {','.join(allowed)}")
    return chosen


def _run(command: list[str]) -> dict[str, str]:
    """@return The parsed fields of one run, or an ``error`` entry."""
    proc = subprocess.run(command, capture_output=True, text=True, cwd=paths.REPO_ROOT)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"error": tail[0][:80]}
    out = proc.stdout
    fields = {}
    for key, pattern in (("speed", _SPEED), ("elapsed", _ELAPSED), ("backend", _BACKEND), ("notes", _NOTES)):
        match = pattern.search(out)
        if match:
            fields[key] = match.group(1)
    if "speed" not in fields:
        return {"error": "could not parse output"}
    return fields


def _cell(fields: dict[str, str]) -> str:
    if "error" in fields:
        return f"— ({fields['error']})"
    return f"{float(fields['speed']):.2f}x ({float(fields['elapsed']):.1f} s, {fields.get('notes', '?')} notes)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--sides", default=",".join(SIDES))
    parser.add_argument("--bench", type=Path, default=paths.CPP_DIR / "build" / "bench" / "muscriptor_bench")
    parser.add_argument("--settle", type=float, default=10.0, help="seconds of idle between runs")
    args = parser.parse_args()

    sizes = _selection(args.sizes, SIZES, "sizes")
    devices = _selection(args.devices, DEVICES, "devices")
    sides = _selection(args.sides, SIDES, "sides")

    if "ggml" in sides and not args.bench.exists():
        sys.exit(f"{args.bench} not built; cmake --build cpp/build")

    # A size without converted weights is skipped up front rather than failing
    # twice on each device. Only the ggml side needs them; torch reads the
    # HuggingFace checkpoint.
    runnable = list(sizes)
    if "ggml" in sides:
        runnable = []
        for size in sizes:
            weights = paths.WEIGHTS_DIR / f"muscriptor-{size}-f16.gguf"
            if weights.exists():
                runnable.append(size)
            else:
                print(f"skipping {size}: {weights} not found", file=sys.stderr)

    results: dict[tuple[str, str, str], dict[str, str]] = {}
    total = len(runnable) * len(devices) * len(sides)
    done = 0
    for size in runnable:
        for device in devices:
            for side in sides:
                done += 1
                label = f"{side} {size}/{device}"
                print(f"[{done}/{total}] {label} ...", file=sys.stderr, flush=True)
                results[(size, device, side)] = _run(_command(side, size, device, args.bench))
                print(f"    {_cell(results[(size, device, side)])}", file=sys.stderr, flush=True)
                if done < total:
                    time.sleep(args.settle)

    columns = [f"{side} {device}" for device in devices for side in sides]
    print("| size | " + " | ".join(columns) + " |")
    print("|---|" + "---|" * len(columns))
    for size in runnable:
        cells = [_cell(results[(size, device, side)]) for device in devices for side in sides]
        print(f"| {size} | " + " | ".join(cells) + " |")
    print()
    print(
        "Whole fixture (15 s, 3 chunks), greedy, prelude forcing on; x = real-time factor, higher is "
        "faster. ggml at F16; torch at its own per-device default, fp16 on mps and fp32 on cpu."
    )
