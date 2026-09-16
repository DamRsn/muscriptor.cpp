"""Download a converted GGUF from the published HuggingFace repo.

A plain HTTP GET against the resolve URL, with no huggingface_hub and no
account.

Only F16 is published. The F32 conversion is a local diagnostic; produce it
with ``msl-convert --weight-dtype f32``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from msl import paths
from msl.checksum import find, sha256
from msl.convert_gguf import FORMAT_VERSION, GGUF_REPO

# One directory per checkpoint format generation.
MODEL_DIR = f"v{FORMAT_VERSION}"
BASE_URL = f"https://huggingface.co/{GGUF_REPO}/resolve/main/{MODEL_DIR}"
CHECKSUMS = "SHA256SUMS"
WEIGHT_DTYPE = "f16"
TIMEOUT_S = 30


def _open(filename: str):
    """GET ``filename`` from the repo, turning HTTP status into a readable error."""
    try:
        return urllib.request.urlopen(f"{BASE_URL}/{filename}", timeout=TIMEOUT_S)
    except urllib.error.HTTPError as error:
        # HuggingFace answers 401 for a repo that is private *or* absent, so
        # the two cases cannot be told apart from here.
        if error.code in (401, 403):
            raise RuntimeError(
                f"{GGUF_REPO} is private or does not exist, so {MODEL_DIR}/{filename} cannot be "
                "downloaded. Convert a checkpoint locally with msl-convert instead."
            ) from error
        if error.code == 404:
            raise RuntimeError(f"{GGUF_REPO} has no {MODEL_DIR}/{filename}") from error
        raise


def _expected_sha256(filename: str) -> str:
    """The repo's own checksum for ``filename``, from its ``SHA256SUMS``."""
    with _open(CHECKSUMS) as response:
        listing = response.read().decode()
    digest = find(listing, filename)
    if digest is None:
        raise LookupError(f"{filename} is not listed in {GGUF_REPO}/{MODEL_DIR}/{CHECKSUMS}")
    return digest


def fetch(size: str, dest: Path | None = None) -> Path:
    """Download the F16 GGUF for ``size`` and verify it. Returns its path."""
    filename = f"muscriptor-{size}-{WEIGHT_DTYPE}.gguf"
    dest = dest or (paths.WEIGHTS_DIR / filename)
    expected = _expected_sha256(filename)

    if dest.exists():
        if sha256(dest) == expected:
            return dest
        raise ValueError(f"{dest} exists and does not match {CHECKSUMS}; delete it to refetch")

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Download beside the target and rename, so an interrupted fetch never
    # leaves a partial file where a complete one is expected.
    partial = dest.with_name(f"{dest.name}.part{os.getpid()}")
    try:
        with _open(filename) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        digest = sha256(partial)
        if digest != expected:
            raise ValueError(f"{filename}: sha256 {digest} != {expected} from {CHECKSUMS}")
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="medium", choices=("small", "medium", "large"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        path = fetch(args.size, args.output)
    except (RuntimeError, LookupError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"{path} ({path.stat().st_size / (1 << 20):.1f} MiB)")
