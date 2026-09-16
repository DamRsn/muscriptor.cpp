"""Filesystem layout.

Everything is anchored on the repo root so the tooling works regardless of the
working directory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TESTDATA_DIR = REPO_ROOT / "testdata"
AUDIO_FIXTURE_DIR = TESTDATA_DIR / "audio"
REFS_DIR = TESTDATA_DIR / "refs"
WEIGHTS_DIR = TESTDATA_DIR / "weights"
# Committed, unlike REFS_DIR: these fixtures are pure integer tables and
# hand-authored token sequences with no model in them, so they cost nothing to
# carry and they are what lets the tokenizer and note-assembly tests run on a
# bare checkout with no weights and no reference dump.
VECTORS_DIR = TESTDATA_DIR / "vectors"

CPP_DIR = REPO_ROOT / "cpp"


def refs_dir(size: str) -> Path:
    """Where the reference dump for one model size lives: ``testdata/refs/<size>``."""
    return REFS_DIR / size


def ensure_dirs() -> None:
    """Create the generated-artifact directories if they don't exist yet."""
    for directory in (AUDIO_FIXTURE_DIR, REFS_DIR, WEIGHTS_DIR, VECTORS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
