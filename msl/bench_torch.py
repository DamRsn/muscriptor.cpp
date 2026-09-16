"""PyTorch-side twin of ``cpp/bench``: times the reference implementation on the
same fixture, chunk, phases and weights, and prints the same FNV-1a token
digest. Matching digests mean both sides decoded the same tokens, so the times
are comparable.

    uv run msl-bench-torch --steps 400              # per-phase breakdown
    uv run msl-bench-torch --transcribe             # whole signal, real-time factor
    uv run msl-bench-torch --device cpu --dtype float32

The reference is driven in place and not patched. The phase split uses
``_model``, ``_device`` and ``_build_conditions``, which are private.

Per-step timing runs yield to yield here, so it includes the reference's argmax
and Python loop overhead; ``cpp/bench`` excludes its argmax but includes the
logits readback.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import muscriptor.accelerator
from muscriptor.transcription_model import TranscriptionModel

from msl import paths

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 80_000

# Matches cpp/bench: FNV-1a over the decoded token ids.
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


def fnv1a(tokens: list[int]) -> int:
    """@return FNV-1a digest of `tokens`, byte-for-byte what cpp/bench prints."""
    digest = _FNV_OFFSET
    for token in tokens:
        digest = ((digest ^ (token & _MASK64)) * _FNV_PRIME) & _MASK64
    return digest


def read_fixture(path: Path) -> torch.Tensor:
    """Read the benchmark fixture, refusing anything that is not what it claims.

    A silently resampled or stereo fixture would make every number below
    meaningless, so this is a check rather than a conversion -- the same
    stance ``readFloatWav`` takes on the C++ side.

    @param path 16 kHz mono float32 WAV.
    @return Samples as a ``[1, T]`` float32 tensor.
    """
    samples, rate = sf.read(path, dtype="float32", always_2d=True)

    if rate != SAMPLE_RATE or samples.shape[1] != 1:
        raise SystemExit(f"expected 16 kHz mono, got rate={rate} ch={samples.shape[1]}")

    return torch.from_numpy(np.ascontiguousarray(samples[:, 0])).unsqueeze(0)


@contextlib.contextmanager
def quiet():
    """Swallow the reference's own progress chatter.

    ``generate`` prints its condition-encoding time to stdout and
    ``transcribe`` prints per-stage timings to stderr. Both are measured here
    with an explicit ``synchronize`` around them, so letting the originals
    through would just interleave two sets of numbers with different fences.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def sync() -> None:
    """Fence the accelerator so a wall-clock reading means something.

    MPS dispatch is asynchronous: without this every phase below would measure
    how fast Python can enqueue work, and the whole cost would pile up on
    whichever call happened to read a tensor back.
    """
    muscriptor.accelerator.synchronize()


def load(args: argparse.Namespace) -> TranscriptionModel:
    """@return The reference model on the requested device and dtype."""
    began = time.perf_counter()
    with quiet():
        model = TranscriptionModel.load_model(
            weights_path=args.model,
            device=args.device,
            dtype=args.dtype,
        )
    sync()
    load_s = time.perf_counter() - began

    dtype = next(model._model.transformer.parameters()).dtype
    print(f"weights   {args.model} ({str(dtype).removeprefix('torch.')})")
    print(f"backend   {str(model._device):>8}")
    print(f"threads   {torch.get_num_threads():>8}")
    print(f"load      {load_s * 1e3:>8.1f} ms")
    return model


def run_phases(model: TranscriptionModel, signal: torch.Tensor, args: argparse.Namespace) -> None:
    """Time conditioning, prefill and per-step decode over one chunk.

    The split follows ``generate``'s own structure: its first yield covers the
    conditioning prefix and the prefill forward, and every later yield is one
    decode step against the KV cache. Conditioning is measured separately by
    running ``condition_provider`` up front -- the same work ``generate`` will
    redo internally -- so prefill can be reported net of it, which is how
    cpp/bench reports it.
    """
    offset = args.chunk * SEGMENT_SAMPLES

    if offset + SEGMENT_SAMPLES > signal.shape[-1]:
        raise SystemExit(f"chunk {args.chunk} is past the end of {signal.shape[-1]} samples")

    chunk = signal[:, offset : offset + SEGMENT_SAMPLES].to(model._device)
    conditions = model._build_conditions(chunk)

    encode_ms: list[float] = []
    prefill_ms: list[float] = []
    decode_ms: list[float] = []
    tokens: list[int] = []

    for _ in range(args.repeats):
        prepared = model._model.condition_provider.tokenize(conditions)
        sync()
        began = time.perf_counter()
        with torch.no_grad(), quiet():
            model._model.condition_provider(prepared)
        sync()
        encode_ms.append((time.perf_counter() - began) * 1e3)

        tokens = []
        step_began = time.perf_counter()

        with quiet():
            stream = model._model.generate(
                conditions=conditions,
                max_gen_len=args.steps,
                use_sampling=False,
                cfg_coef=1.0,
                # None, not the real EOS: cpp/bench decodes a fixed step count
                # and ignores EOS, and a short chunk that stopped early would
                # silently benchmark fewer steps than asked for.
                early_stop_on_token=None,
            )

            for step in stream:
                token = int(step[0])
                sync()
                elapsed_ms = (time.perf_counter() - step_began) * 1e3

                # The first yield carries the conditioning prefix and the
                # prefill forward; everything after it is a single decode.
                (prefill_ms if not tokens else decode_ms).append(elapsed_ms)
                tokens.append(token)
                step_began = time.perf_counter()

    # generate() re-encodes the conditions internally, so its first yield
    # includes work already accounted for above. Report prefill net of it.
    #
    # Subtract the best encode, the smallest correction; report the clamp if it
    # ever fires.
    prefill_net = [max(0.0, p - min(encode_ms)) for p in prefill_ms]

    if any(p - min(encode_ms) < 0.0 for p in prefill_ms):
        print(
            "warning: encode time exceeded a measured prefill, so the split between "
            "the two is unreliable for this run",
            file=sys.stderr,
        )

    ordered = sorted(decode_ms)
    mean = statistics.fmean(decode_ms) if decode_ms else 0.0

    def percentile(values: list[float], q: float) -> float:
        return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0

    print(f"encode    {min(encode_ms):>8.1f} ms   (best of {len(encode_ms)})")
    print(f"prefill   {min(prefill_net):>8.1f} ms   (best of {len(prefill_net)})")
    print(
        f"decode    {mean:>8.2f} ms/step   min {min(ordered):.2f}  "
        f"p50 {percentile(ordered, 0.50):.2f}  p90 {percentile(ordered, 0.90):.2f}  "
        f"max {ordered[-1]:.2f}   over {len(ordered)} steps"
    )
    print(f"decode    {1000.0 / mean if mean else 0.0:>8.1f} tok/s")

    head = " ".join(str(t) for t in tokens[:24])
    print(f"tokens    {head}{' ...' if len(tokens) > 24 else ''}")
    print(f"digest    {fnv1a(tokens):016x}  ({len(tokens)} tokens)")


def run_transcribe(model: TranscriptionModel, signal: torch.Tensor, args: argparse.Namespace) -> None:
    """Time the whole signal end to end -- the number a caller actually feels.

    Greedy, ``batch_size=1``, prelude forcing on: the defaults of both the
    reference and ``Transcriber``, so the two are comparable.
    """
    from muscriptor.events import NoteStartEvent

    sync()
    began = time.perf_counter()
    with quiet():
        events = list(
            model.transcribe(
                (signal, SAMPLE_RATE),
                use_sampling=False,
                batch_size=1,
                prelude_forcing=True,
            )
        )
    sync()
    elapsed_s = time.perf_counter() - began

    audio_s = signal.shape[-1] / SAMPLE_RATE
    chunks = -(-signal.shape[-1] // SEGMENT_SAMPLES)
    notes = sum(1 for e in events if isinstance(e, NoteStartEvent))

    print(f"audio     {audio_s:>8.2f} s   ({chunks} chunks)")
    print(f"transcribe {elapsed_s:>7.2f} s")
    print(f"speed     {audio_s / elapsed_s:>8.2f}x real time")
    print(f"notes     {notes:>8d}")


def main() -> None:
    default_device = "mps" if muscriptor.accelerator.is_available() else "cpu"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, default=paths.AUDIO_FIXTURE_DIR / "fixture_3chunks_16k.wav")
    parser.add_argument("--model", "--size", default="medium", help="size keyword or weights path")
    parser.add_argument("--device", default=default_device, choices=("mps", "cpu", "cuda"))
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float16", "float32", "bfloat16"),
        help="default: the reference's own per-device choice (fp16 on MPS, fp32 elsewhere)",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0, help="CPU thread pool; 0 leaves torch's default")
    parser.add_argument("--transcribe", action="store_true", help="whole signal instead of the phase breakdown")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    signal = read_fixture(args.audio)
    model = load(args)

    if args.transcribe:
        run_transcribe(model, signal, args)
    else:
        run_phases(model, signal, args)


if __name__ == "__main__":
    sys.exit(main())
