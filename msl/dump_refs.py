"""Run the reference PyTorch model and dump every stage for the C++ suite.

Produces, in ``testdata/refs/<size>/``:

* ``refs_fp32.gguf`` -- reference with fp32 weights and activations.
* ``refs_fp16.gguf`` -- reference with the model ``.half()``-ed, i.e. the
  precision muscriptor actually runs at on Apple Silicon.
* ``manifest.json`` -- shapes, token sequences, provenance, and the tolerances
  the C++ tests assert against.
* ``notes.json`` -- the note-level variants.

Why two references: ggml keeps *activations* in F32 and only stores *weights* as
F16, whereas ``model.half()`` puts activations in fp16 too. Neither is "the"
answer, so the fp32 run is the comparison target and the fp32<->fp16 gap is
measured and published as the noise floor. A port that lands inside that gap is
as faithful as the sanctioned half-precision path already is -- which makes the
tolerances a measurement rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from msl import audio_fixture, convert_gguf, environment, note_refs, paths, tensor_io
from msl.hooks import Tracer

ARCH = "muscriptor"
MAX_GEN_LEN = 2000
DEFAULT_MAX_DECODE_STEPS = 16
PRECISIONS = ("fp32", "fp16")
_TORCH_DTYPE = {"fp32": "float32", "fp16": "float16"}


# ---------------------------------------------------------------------------
# Reference run
# ---------------------------------------------------------------------------


def _generate_tokens(lm, conditions, eos_id: int) -> list[int]:
    """Greedy-decode one chunk, returning tokens up to and including EOS.

    Mirrors ``TranscriptionModel._generate_token_stream`` with ``batch_size=1``
    and ``prelude_forcing=False``: no prompt, no tie prologue, chunks fully
    independent of each other.
    """
    tokens: list[int] = []
    for step in lm.generate(
        prompt=None,
        conditions=conditions,
        max_gen_len=MAX_GEN_LEN,
        use_sampling=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        cfg_coef=1.0,
        early_stop_on_token=eos_id,
        beam_size=1,
        forbidden_tokens=None,
    ):
        token = int(step[0].item())
        tokens.append(token)
        if token == eos_id:
            break
    return tokens


def run_reference(
    size: str,
    precision: str,
    num_chunks: int,
    max_decode_steps: int,
    variants: tuple[note_refs.Variant, ...] = (),
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    """Load the model at ``precision`` and trace chunk 0 plus decode all chunks.

    Returns the tensors, the run info, and the model's mel filterbank.

    ``variants`` adds note-level runs on top of the token-level ladder: the same
    fixture decoded with prelude forcing on, and with instrument selection. The
    ``tokens`` block stays unconditional and unforced, so the chunks are
    independently checkable.
    """
    from muscriptor.transcription_model import TranscriptionModel

    model = TranscriptionModel.load_model(
        weights_path=size, device="cpu", dtype=_TORCH_DTYPE[precision]
    )
    lm = model._model
    eos_id = model._tokenizer.eos_id
    num_layers = len(lm.transformer.layers)

    tracer = Tracer()
    tokens: dict[str, list[int]] = {}
    all_chunks = audio_fixture.chunks(num_chunks)
    for index, chunk in enumerate(all_chunks):
        wav = torch.from_numpy(chunk).unsqueeze(0)  # [1, segment_samples]
        conditions = model._build_conditions(wav, instrument_group=None)
        started = time.perf_counter()
        if index == 0:
            with tracer.attach(lm, num_layers):
                tokens["chunk0"] = _generate_tokens(lm, conditions, eos_id)
        else:
            tokens[f"chunk{index}"] = _generate_tokens(lm, conditions, eos_id)
        print(
            f"  chunk{index}: {len(tokens[f'chunk{index}'])} tokens "
            f"in {time.perf_counter() - started:.1f}s"
        )

    tensors = tracer.finalize(max_decode_steps)
    # Per-chunk waveforms, for the STFT and Transcriber tests.
    for index, chunk in enumerate(all_chunks):
        tensors[f"in.chunk{index}.wav"] = chunk.astype(np.float32)

    # Spectra for every chunk, not just the traced one, so later chunks can be
    # conditioned from the dump and show the KV cache resets between chunks.
    mel_module = lm.condition_provider.conditioners["self_wav"].mel_spec_transform
    for index, chunk in enumerate(all_chunks):
        spec = torch.stft(
            torch.from_numpy(chunk).unsqueeze(0),
            n_fft=mel_module.n_fft,
            hop_length=mel_module.hop_length,
            win_length=mel_module.n_fft,
            window=mel_module.spectrogram.window.to(torch.float32),
            center=mel_module.center,
            pad_mode=mel_module.pad_mode,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        tensors[f"in.chunk{index}.spectrum"] = (
            spec.abs().transpose(-1, -2)[0].to(torch.float32).numpy()
        )

    # Recomputing chunk 0 must reproduce what the hook captured during the real
    # forward pass; if it does not, this loop is not the model's STFT.
    if not np.allclose(tensors["in.chunk0.spectrum"], tensors["in.spectrum"]):
        raise AssertionError(
            "recomputed chunk-0 spectrum differs from the one captured in the forward pass"
        )

    # The filterbank the model actually used. Note this is the buffer stored in
    # the checkpoint (generated by torchaudio), which differs from muscriptor's
    # pure-torch `melscale_fbanks` reimplementation by ~2e-4 relative -- so the
    # checkpoint's copy is the only correct thing to validate against, and it is
    # what the converter exports.
    mel_fb = (
        lm.condition_provider.conditioners["self_wav"]
        .mel_spec_transform.mel_scale.fb.detach()
        .to(torch.float32)
        .numpy()
    )

    # Note-level runs. Each is a full extra pass over the fixture, so they cost
    # roughly one `plain` run each -- filter with --variants while iterating.
    variant_results: dict[str, Any] = {}
    for variant in variants:
        started = time.perf_counter()
        result = note_refs.run_variant(model, variant, all_chunks)
        note_refs.check_variant(result)
        variant_results[variant.name] = result
        print(
            f"  variant {variant.name}: {sum(len(t) for t in result['tokens'])} tokens, "
            f"{len(result['notes'])} notes in {time.perf_counter() - started:.1f}s"
        )

        if variant.trace_prefix:
            traced = note_refs.trace_prefix(model, variant, all_chunks[0])
            tensors.update(traced["tensors"])
            result["prefix"] = traced["info"]

    info = {
        "model": {
            "size": size,
            "dim": lm.dim,
            "num_heads": lm.transformer.layers[0].self_attn.num_heads,
            "head_dim": lm.transformer.layers[0].self_attn.dim_per_head,
            "num_layers": num_layers,
            "ffn_dim": lm.transformer.layers[0].linear1.out_features,
            "card": lm.card,
            "initial_token_id": lm.initial_token_id,
            "eos_id": eos_id,
            "layer_norm_eps": convert_gguf.LAYER_NORM_EPS,
            "max_period": convert_gguf.MAX_PERIOD,
        },
        "tokens": tokens,
        "decode_argmax": {
            name.removesuffix(".logits"): int(np.argmax(value))
            for name, value in sorted(tensors.items())
            if name.startswith("dec.step")
        },
        "variants": variant_results,
    }
    return tensors, info, mel_fb


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def _check_consistency(tensors: dict[str, np.ndarray], fb: np.ndarray) -> None:
    """Assert the dumped stages actually chain together.

    Guards the layout convention and the hook wiring: if ``in.spectrum`` were
    transposed, or ``cond.mel`` came from the wrong hook, the C++ suite would
    chase a phantom bug instead of failing here.
    """
    spectrum, mel, logmel = tensors["in.spectrum"], tensors["cond.mel"], tensors["cond.logmel"]
    recomputed = spectrum.astype(np.float64) @ fb.astype(np.float64)
    error = np.abs(recomputed - mel).max() / max(np.abs(mel).max(), 1e-12)
    if error > 1e-5:
        raise AssertionError(
            f"in.spectrum @ mel_fb does not reproduce cond.mel (rel err {error:.2e}); "
            "the recorded layouts are inconsistent"
        )
    if not np.allclose(logmel, np.log(mel + 1e-6), rtol=1e-5, atol=1e-6):
        raise AssertionError("cond.logmel is not log(cond.mel + 1e-6)")


def _check_variants(info: dict[str, Any]) -> None:
    """Cross-check the note-level runs against the token-level ladder.

    The ``plain`` variant decodes the same configuration the ``tokens`` block
    already records, so the two must agree token for token -- modulo EOS, which
    ``_generate_tokens`` keeps and the stream layer strips. If they disagree,
    the two dumps are not describing the same run and neither is trustworthy.
    """
    variants = info.get("variants", {})
    plain = variants.get("plain")

    if plain is None:
        return

    for index, chunk_tokens in enumerate(plain["tokens"]):
        ladder = info["tokens"][f"chunk{index}"]
        expected = ladder[:-1] if ladder and ladder[-1] == info["model"]["eos_id"] else ladder
        if chunk_tokens != expected:
            raise AssertionError(
                f"the plain variant and the token ladder disagree on chunk {index}: "
                f"{len(chunk_tokens)} vs {len(expected)} tokens"
            )

    # Prelude forcing must change the token streams. If it does not, either it
    # is not actually enabled or no note ever crosses a boundary, and in both
    # cases the fixture cannot validate the feature it exists for.
    prelude = variants.get("prelude")
    if prelude is not None and prelude["tokens"] == plain["tokens"]:
        raise AssertionError(
            "prelude forcing produced identical token streams to the unforced run; "
            "the fixture cannot validate it"
        )


# ---------------------------------------------------------------------------
# Tolerance calibration
# ---------------------------------------------------------------------------


def _stats(name: str, reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Compare two references, ignoring entries that are non-finite in both.

    ``_compute_logits`` forces the reserved token ids to ``-inf``, so the logits
    tensors are legitimately non-finite in exactly the masked positions. Those
    positions must agree between the two runs (a differing mask is a real bug),
    but ``-inf - -inf`` is NaN, so they are excluded from the numeric stats.
    """
    a = reference.astype(np.float64).ravel()
    b = other.astype(np.float64).ravel()

    finite = np.isfinite(a)
    if not np.array_equal(finite, np.isfinite(b)):
        raise AssertionError(f"{name}: fp32 and fp16 differ in which entries are finite")
    if not np.array_equal(a[~finite], b[~finite]):
        raise AssertionError(f"{name}: fp32 and fp16 differ in their non-finite values")
    a, b = a[finite], b[finite]

    diff = np.abs(a - b)
    scale = max(float(np.abs(a).max()), 1e-12)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "max_abs": float(diff.max()),
        "max_rel": float(diff.max() / scale),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        "cosine": float(np.dot(a, b) / denom) if denom > 0 else 1.0,
        "ref_absmax": scale,
    }


# How far past the measured fp32<->fp16 gap the ggml port is allowed to sit.
# If a stage misses it, investigate; do not widen this number.
NOISE_FLOOR_MULTIPLE = 4.0


def _tolerances(noise_floor: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Turn the measured fp32<->fp16 gap into per-tensor pass thresholds.

    Floored at 1e-5 of the tensor's own scale so that stages where the two
    references are bit-identical -- the whole conditioning path, which stays
    fp32 in both -- still get a tight, non-zero bound instead of demanding
    exactness the port cannot deliver.
    """
    out = {}
    for name, stats in noise_floor.items():
        floor = 1e-5 * stats["ref_absmax"]
        out[name] = {
            "atol": max(NOISE_FLOOR_MULTIPLE * stats["max_abs"], floor),
            "cosine_min": 1.0
            - max(NOISE_FLOOR_MULTIPLE * (1.0 - stats["cosine"]), 1e-7),
        }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def dump(
    size: str,
    num_chunks: int = audio_fixture.DEFAULT_NUM_CHUNKS,
    max_decode_steps: int = DEFAULT_MAX_DECODE_STEPS,
    variants: tuple[note_refs.Variant, ...] = note_refs.VARIANTS,
) -> Path:
    paths.ensure_dirs()
    refs_dir = paths.refs_dir(size)
    refs_dir.mkdir(parents=True, exist_ok=True)
    fixture_file = audio_fixture.fixture_path(num_chunks)
    fixture_info = audio_fixture.describe(num_chunks)
    print(f"fixture: {fixture_file}")

    references: dict[str, dict[str, np.ndarray]] = {}
    infos: dict[str, dict[str, Any]] = {}
    for precision in PRECISIONS:
        print(f"running reference: {size} / {precision} ...")
        tensors, info, mel_fb = run_reference(
            size, precision, num_chunks, max_decode_steps, variants
        )
        _check_consistency(tensors, mel_fb)
        _check_variants(info)
        out = refs_dir / f"refs_{precision}.gguf"
        tensor_io.write_gguf(
            out,
            ARCH,
            tensors,
            metadata={
                f"{ARCH}.precision": precision,
                f"{ARCH}.size": size,
                f"{ARCH}.tensor_count": len(tensors),
            },
        )
        print(f"  wrote {out} ({len(tensors)} tensors)")
        references[precision] = tensors
        infos[precision] = info

    shared = sorted(set(references["fp32"]) & set(references["fp16"]))
    noise_floor = {
        name: _stats(name, references["fp32"][name], references["fp16"][name])
        for name in shared
    }

    tokens_match = infos["fp32"]["tokens"] == infos["fp16"]["tokens"]
    print(
        f"\ntoken streams agree across precisions: {tokens_match}"
        + ("" if tokens_match else "  (see manifest['tokens'] for both)")
    )

    manifest = {
        "environment": environment.describe(),
        "fixture": asdict(fixture_info),
        "model": infos["fp32"]["model"],
        "generation": {
            "greedy": True,
            "prelude_forcing": False,
            "max_gen_len": MAX_GEN_LEN,
            "num_chunks": num_chunks,
            "max_decode_steps_dumped": max_decode_steps,
        },
        # Each condition is cat-ed in front of the previous one, so the emitted
        # order is the reverse of ConditioningProvider's iteration order. These
        # are the unconditional numbers (one null-class instrument row); the
        # per-variant blocks carry the conditioned lengths.
        "prefix": {
            "order": ["self_wav", "dataset_name", "instrument_group"],
            "self_wav_frames": int(references["fp32"]["cond.embed"].shape[0]),
            "dataset_name_frames": 1,
            "instrument_group_frames": int(
                references["fp32"]["cond.instrument_group"].shape[0]
            ),
            "prepend_length": int(references["fp32"]["cond.embed"].shape[0])
            + 1
            + int(references["fp32"]["cond.instrument_group"].shape[0]),
        },
        "variants": _variant_summary(infos),
        "references": {p: f"refs_{p}.gguf" for p in PRECISIONS},
        "tensors": {
            name: {
                "shape": list(references["fp32"][name].shape),
                "dtype": str(references["fp32"][name].dtype),
            }
            for name in shared
        },
        "tokens": {p: infos[p]["tokens"] for p in PRECISIONS},
        # When true, greedy decoding is insensitive to the fp32/fp16 difference
        # over these chunks, so demanding token-exactness from the port is a
        # reasonable bar rather than a coin flip on near-ties.
        "tokens_match_across_precisions": tokens_match,
        "decode_argmax": {p: infos[p]["decode_argmax"] for p in PRECISIONS},
        "noise_floor": noise_floor,
        "tolerances": _tolerances(noise_floor),
    }
    manifest_path = refs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {manifest_path}")

    # The bulky per-variant data lives in its own file: actions, events and two
    # note lists for every variant and both precisions runs to megabytes, and
    # manifest.json is read by every test case in the suite.
    notes_path = refs_dir / "notes.json"
    notes_path.write_text(
        json.dumps({p: infos[p]["variants"] for p in PRECISIONS}, indent=2, sort_keys=True)
    )
    print(f"wrote {notes_path}")

    _print_summary(noise_floor)
    _print_variant_summary(manifest["variants"])
    return manifest_path


def _variant_summary(infos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-variant configuration and shape, small enough for the manifest."""
    summary: dict[str, Any] = {}

    for name, result in infos["fp32"].get("variants", {}).items():
        other = infos["fp16"].get("variants", {}).get(name)
        summary[name] = {
            "prelude_forcing": result["prelude_forcing"],
            "instruments": result["instruments"],
            "instrument_group_string": result["instrument_group_string"],
            "conditioning_rows": result["conditioning_rows"],
            "forbidden_token_count": len(result["forbidden_token_ids"]),
            "note_count": len(result["notes"]),
            "token_counts": [len(chunk) for chunk in result["tokens"]],
            "tokens_match_across_precisions": (
                other is not None and other["tokens"] == result["tokens"]
            ),
        }

        if "prefix" in result:
            summary[name]["prefix"] = result["prefix"]

    return summary


def _print_summary(noise_floor: dict[str, dict[str, float]], limit: int = 12) -> None:
    ranked: Iterable[tuple[str, dict[str, float]]] = sorted(
        noise_floor.items(), key=lambda kv: kv[1]["cosine"]
    )
    print("\nfp32 vs fp16 noise floor (worst cosine first):")
    print(f"  {'tensor':<26} {'max_abs':>10} {'max_rel':>10} {'cosine':>12}")
    for name, stats in list(ranked)[:limit]:
        print(
            f"  {name:<26} {stats['max_abs']:>10.3e} {stats['max_rel']:>10.3e} "
            f"{stats['cosine']:>12.9f}"
        )


def _print_variant_summary(variants: dict[str, Any]) -> None:
    if not variants:
        return

    print("\nnote-level variants:")
    print(f"  {'variant':<12} {'forcing':>8} {'rows':>5} {'forbidden':>10} {'notes':>7}  instruments")
    for name, info in variants.items():
        print(
            f"  {name:<12} {str(info['prelude_forcing']):>8} "
            f"{len(info['conditioning_rows']):>5} {info['forbidden_token_count']:>10} "
            f"{info['note_count']:>7}  {', '.join(info['instruments']) or '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="medium", choices=("small", "medium", "large"))
    parser.add_argument("--num-chunks", type=int, default=audio_fixture.DEFAULT_NUM_CHUNKS)
    parser.add_argument("--max-decode-steps", type=int, default=DEFAULT_MAX_DECODE_STEPS)
    parser.add_argument(
        "--variants",
        default=",".join(v.name for v in note_refs.VARIANTS),
        help=(
            "comma-separated note-level variants to run, or 'none'. Each is a "
            "full extra pass over the fixture, so narrow this while iterating."
        ),
    )
    args = parser.parse_args()

    selected = (
        ()
        if args.variants.strip().lower() == "none"
        else tuple(
            note_refs.variant_by_name(name.strip())
            for name in args.variants.split(",")
            if name.strip()
        )
    )
    dump(args.size, args.num_chunks, args.max_decode_steps, selected)
