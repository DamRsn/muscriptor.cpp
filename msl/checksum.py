"""The ``SHA256SUMS`` format the published GGUF repo ships alongside its files.

Both sides of the round trip live here: the converter prints a digest to fill
the file with, and the fetcher parses it back to verify a download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# `sha256sum` writes two spaces between the digest and the name.
SEPARATOR = "  "


def sha256(path: Path) -> str:
    """Hex sha256 of a file, the form ``SHA256SUMS`` lists."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def find(listing: str, filename: str) -> str | None:
    """The digest ``listing`` records for ``filename``, or ``None``."""
    for line in listing.splitlines():
        digest, _, name = line.partition(SEPARATOR)
        if name.strip() == filename:
            return digest.strip()
    return None
