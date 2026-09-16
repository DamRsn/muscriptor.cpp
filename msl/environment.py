"""Capture the exact toolchain a set of artifacts was produced with.

Every reference dump embeds this dict in its manifest, so a dumped number can be
traced back to the ggml commit, reference implementation and compiler.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path

from msl import paths


def _git_commit(repo: Path) -> str:
    """Short commit hash of ``repo``, suffixed with ``-dirty`` if it has
    uncommitted changes. ``"unknown"`` when the directory isn't a git repo."""
    if not (repo / ".git").exists():
        return "unknown"
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _muscriptor_commit() -> str:
    """Short commit of the installed ``muscriptor`` distribution, from its
    ``direct_url.json`` record."""
    from importlib.metadata import Distribution

    record = Distribution.from_name("muscriptor").read_text("direct_url.json")
    if record is None:
        return "unknown"
    return json.loads(record).get("vcs_info", {}).get("commit_id", "unknown")[:8]


def _pinned_ggml_commit() -> str:
    """Short form of the ggml commit ``cpp/CMakeLists.txt`` pins.

    Read from the pin, which is the commit the build fetches.
    """
    path = paths.CPP_DIR / "CMakeLists.txt"
    match = re.search(r"set\(MUSCRIPTOR_GGML_COMMIT\s+([0-9a-f]{40})\)", path.read_text())
    if match is None:
        raise RuntimeError(f"no MUSCRIPTOR_GGML_COMMIT pin found in {path}")
    return match.group(1)[:8]


def _tool_version(argv: list[str], line: int = 0) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return "unavailable"
    text = out.stdout or out.stderr
    lines = text.splitlines()
    return lines[line].strip() if len(lines) > line else text.strip()


def describe() -> dict[str, str]:
    """A flat, JSON-serialisable description of the build/reference toolchain."""
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": _tool_version(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "muscriptor_commit": _muscriptor_commit(),
        "ggml_commit": _pinned_ggml_commit(),
        "muscriptor_cpp_commit": _git_commit(paths.REPO_ROOT),
        "clang": _tool_version(["clang++", "--version"]),
        "cmake": _tool_version(["cmake", "--version"]),
    }
