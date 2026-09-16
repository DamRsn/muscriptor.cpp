"""The committed audio fixture, ``testdata/audio/fixture_3chunks_16k.wav``.

``msl-audio-fixture --source <mp3>`` rebuilds it: the source is decoded with the
reference implementation's own loader (``muscriptor.utils.audio.load_audio``,
which resamples to 16 kHz with the filter the model was evaluated with), sliced,
and written as a float32 WAV.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from msl import paths
from msl.checksum import sha256

SAMPLE_RATE = 16_000
SEGMENT_DURATION = 5.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)

# "You Drive Me Insane" by Jon Worthy and the Bends, CC BY 4.0 (attribution in
# testdata/audio/README.md). Its opening has distorted guitar, bass and drums
# throughout, a synth lead in the middle chunk and vocals from about 12 s. Three
# chunks is the smallest slice that catches state that fails to reset between
# chunks.
DEFAULT_START_SECONDS = 0.0
DEFAULT_NUM_CHUNKS = 3


@dataclass(frozen=True)
class FixtureInfo:
    """Description of the fixture, embedded in the reference manifest."""

    num_chunks: int
    sample_rate: int
    segment_samples: int
    total_samples: int
    fixture_sha256: str


def fixture_path(num_chunks: int = DEFAULT_NUM_CHUNKS) -> Path:
    return paths.AUDIO_FIXTURE_DIR / f"fixture_{num_chunks}chunks_16k.wav"


def _has_same_samples(path: Path, samples: np.ndarray) -> bool:
    """True if `path` already holds exactly these samples at this rate."""
    if not path.exists():
        return False

    try:
        existing, rate = sf.read(path, dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        return False

    return rate == SAMPLE_RATE and np.array_equal(existing, samples)


def build(
    source: Path,
    start_seconds: float = DEFAULT_START_SECONDS,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
) -> Path:
    """Decode, slice and write the fixture from `source`. Returns its path."""
    from muscriptor.utils.audio import load_audio

    paths.ensure_dirs()
    if not source.exists():
        raise FileNotFoundError(f"source audio not found: {source}")

    wav = load_audio(source, target_sr=SAMPLE_RATE)  # [1, T] mono float32
    start = int(start_seconds * SAMPLE_RATE)
    total = num_chunks * SEGMENT_SAMPLES
    if wav.shape[-1] < start + total:
        raise ValueError(
            f"{source.name} is too short: need {start + total} samples at "
            f"{SAMPLE_RATE} Hz, have {wav.shape[-1]}"
        )
    sliced = wav[0, start : start + total].contiguous().numpy().astype(np.float32)

    out = fixture_path(num_chunks)

    # PCM_FLOAT keeps the samples bit-exact. Written only when the samples
    # differ, since libsndfile stamps its own version into the file.
    if not _has_same_samples(out, sliced):
        sf.write(out, sliced, SAMPLE_RATE, subtype="FLOAT")

    return out


def describe(num_chunks: int = DEFAULT_NUM_CHUNKS) -> FixtureInfo:
    """Sizes and checksum of the committed fixture."""
    return FixtureInfo(
        num_chunks=num_chunks,
        sample_rate=SAMPLE_RATE,
        segment_samples=SEGMENT_SAMPLES,
        total_samples=int(load(num_chunks).shape[0]),
        fixture_sha256=sha256(fixture_path(num_chunks)),
    )


def load(num_chunks: int = DEFAULT_NUM_CHUNKS) -> np.ndarray:
    """Read the fixture as float32 [total_samples]."""
    path = fixture_path(num_chunks)
    if not path.exists():
        raise FileNotFoundError(f"audio fixture not found: {path}")
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"fixture sample rate is {rate}, expected {SAMPLE_RATE}")
    return np.ascontiguousarray(data)


def chunks(num_chunks: int = DEFAULT_NUM_CHUNKS) -> list[np.ndarray]:
    """The fixture split into the model's 5-second segments."""
    data = load(num_chunks)
    return [
        data[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES] for i in range(num_chunks)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=DEFAULT_START_SECONDS)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    args = parser.parse_args()

    path = build(args.source, args.start_seconds, args.num_chunks)
    print(f"wrote {path}")
    for key, value in asdict(describe(args.num_chunks)).items():
        print(f"  {key}: {value}")
