"""Note-level references from real model output, for every decode configuration.

The token-level dump uses one configuration, prelude forcing off and no
instrument filter, so its chunks are independently checkable. These variants
carry decoding down to actions, events and notes:

``plain``       forcing off, no filter: the token-level configuration.
``prelude``     forcing on, the upstream and ``Transcriber`` default. Records
                the forced prologue for every chunk separately from the tokens,
                so a mismatch localises to the prompt rather than to the
                transformer.
``bass``        one instrument selected. N = 1 conditioning row.
``band``        five, one of which is drums. N = 5, which catches a prefix
                length hardcoded to "one row".

Each variant is driven through ``TranscriptionModel._generate_token_stream`` --
upstream's own driver, not a re-derivation of it here. That matters most for
prelude forcing: the thing being ported is precisely the order in which the
boundary is fed, the open keys are read, and the prompt is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from msl import note_decode

MAX_GEN_LEN = 2000
SEGMENT_DURATION = 5.0


@dataclass(frozen=True)
class Variant:
    """One decode configuration to dump."""

    name: str
    prelude_forcing: bool
    instruments: tuple[str, ...]
    # Capture the assembled prefix and the instrument-group embedding rows for
    # this variant. Only worth it where the conditioning segment differs from
    # the unconditional case the ladder already covers.
    trace_prefix: bool


VARIANTS: tuple[Variant, ...] = (
    Variant("plain", prelude_forcing=False, instruments=(), trace_prefix=False),
    Variant("prelude", prelude_forcing=True, instruments=(), trace_prefix=False),
    Variant(
        "bass",
        prelude_forcing=False,
        instruments=("electric_bass",),
        trace_prefix=True,
    ),
    Variant(
        "band",
        prelude_forcing=False,
        instruments=(
            "distorted_electric_guitar",
            "synth_lead",
            "electric_bass",
            "drums",
            "voice",
        ),
        trace_prefix=True,
    ),
)


def variant_by_name(name: str) -> Variant:
    for variant in VARIANTS:
        if variant.name == name:
            return variant

    raise KeyError(f"unknown variant {name!r}; have {[v.name for v in VARIANTS]}")


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def instrument_group_string(instruments: tuple[str, ...]) -> str | None:
    """The space-separated group-id string the ClassConditioner tokenizes.

    None for an empty selection, which selects the null class -- exactly one
    prefix row. A non-empty selection yields one row per id.
    """
    from muscriptor.tokenizer.mt3 import instrument_group_from_names

    return instrument_group_from_names(instruments) if instruments else None


def conditioning_rows(instruments: tuple[str, ...]) -> list[int]:
    """Embedding rows the instrument_group conditioner reads for `instruments`.

    ``ClassConditioner.tokenize`` adds one and ``forward`` adds one again, so a
    group id lands on row ``gid + 2`` and the null class on row 1. Spelled out
    here because the double offset reads like an off-by-one at a glance.
    """
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES

    if not instruments:
        return [1]

    return [MT3_FULL_PLUS_GROUP_NAMES[name] + 2 for name in instruments]


def forbidden_token_ids(tokenizer: Any, instruments: tuple[str, ...]) -> list[int]:
    """The hard instrument mask, or nothing at all for an empty selection.

    Note the guard: ``forbidden_token_ids([])`` forbids every program and every
    drum token, so calling it unconditionally would turn "no filter" into "no
    instruments". Upstream guards the same way (`transcribe` leaves
    ``forbidden_tokens`` as None), and so must the port.
    """
    if not instruments:
        return []

    return list(tokenizer.forbidden_token_ids(instruments))


# ---------------------------------------------------------------------------
# Running one variant
# ---------------------------------------------------------------------------


def run_variant(
    model: Any,
    variant: Variant,
    chunks: list[np.ndarray],
) -> dict[str, Any]:
    """Transcribe the fixture under `variant`, recording every level.

    @param model A loaded ``TranscriptionModel``.
    @param variant The decode configuration.
    @param chunks The fixture's 5-second segments, already 80000 samples each.
    @return Per-chunk tokens and forced prompts, plus actions, events and notes.
    """
    from muscriptor.events import ChunkBoundary
    from muscriptor.transcription_model import _build_instrument_for_program

    tokenizer = model._tokenizer
    group_string = instrument_group_string(variant.instruments)
    seek_times = [index * SEGMENT_DURATION for index in range(len(chunks))]

    all_conditions = [
        model._build_conditions(
            torch.from_numpy(chunk).unsqueeze(0), instrument_group=group_string
        )[0]
        for chunk in chunks
    ]

    forbidden = forbidden_token_ids(tokenizer, variant.instruments)
    forbidden_tensor = (
        torch.tensor(forbidden, device=model._device, dtype=torch.long)
        if forbidden
        else None
    )

    # Record the forced prologues as the reference builds them, rather than
    # rebuilding them here from a tracker of our own. Re-deriving would mean the
    # fixture and the thing it validates share a bug.
    prompts: list[dict[str, Any]] = []
    original_tie_section = tokenizer.tie_section_token_ids

    def spy(open_note_keys):
        keys = list(open_note_keys)
        token_ids = original_tie_section(keys)
        prompts.append(
            {
                "open_keys": [list(key) for key in keys],
                "token_ids": list(token_ids),
            }
        )
        return token_ids

    tokenizer.tie_section_token_ids = spy
    try:
        items = [
            item
            for item in model._generate_token_stream(
                all_conditions=all_conditions,
                seek_times=seek_times,
                batch_size=1,
                max_gen_len=MAX_GEN_LEN,
                use_sampling=False,
                temperature=1.0,
                cfg_coef=1.0,
                no_eos_is_ok=True,
                prelude_forcing=variant.prelude_forcing,
                beam_size=1,
                forbidden_tokens=forbidden_tensor,
            )
            if isinstance(item, (int, ChunkBoundary))
        ]
    finally:
        tokenizer.tie_section_token_ids = original_tie_section

    # Split the flat stream back into chunks. The tokens recorded here have EOS
    # already stripped by the stream layer, and they *include* the teacher-forced
    # prompt -- which is what Model::generate returns too, so the C++ comparison
    # is against the same thing.
    chunk_tokens: list[list[int]] = []
    for item in items:
        if isinstance(item, ChunkBoundary):
            chunk_tokens.append([])
        else:
            chunk_tokens[-1].append(int(item))

    decoded = note_decode.decode(
        iter(items), tokenizer, _build_instrument_for_program(tokenizer)
    )
    replay = note_decode.run_tracker(items, tokenizer)

    # Chunk 0 is never forced, so the recorded prompts line up with chunks 1..n.
    per_chunk_prompts: list[list[int]] = [[]]
    per_chunk_prompts.extend(prompt["token_ids"] for prompt in prompts)
    per_chunk_prompts += [[]] * (len(chunk_tokens) - len(per_chunk_prompts))

    return {
        "name": variant.name,
        "prelude_forcing": variant.prelude_forcing,
        "instruments": list(variant.instruments),
        "instrument_group_string": group_string,
        "conditioning_rows": conditioning_rows(variant.instruments),
        "forbidden_token_ids": forbidden,
        "seek_times": seek_times,
        "tokens": chunk_tokens,
        "prompts": per_chunk_prompts,
        "open_keys_at_boundary": replay["open_keys"],
        **decoded,
    }


def check_variant(result: dict[str, Any]) -> None:
    """Cross-check one variant's recorded levels against each other."""
    name = result["name"]
    note_decode.check(result, name)

    if result["prelude_forcing"]:
        # Every forced prologue must be exactly what the open keys at that
        # boundary encode to. If this fails, the two halves of prelude forcing
        # have already diverged inside the reference and the fixture is not
        # worth porting against.
        for index, prompt in enumerate(result["prompts"][1:], start=1):
            if not prompt:
                raise AssertionError(f"{name}: chunk {index} has no forced prologue")

            if result["tokens"][index][: len(prompt)] != prompt:
                raise AssertionError(
                    f"{name}: chunk {index}'s token stream does not start with its "
                    "forced prologue"
                )
    elif any(result["prompts"]):
        raise AssertionError(f"{name}: forcing is off but a prologue was forced")

    if result["forbidden_token_ids"]:
        forbidden = set(result["forbidden_token_ids"])
        for index, tokens in enumerate(result["tokens"]):
            hit = forbidden.intersection(tokens)
            if hit:
                raise AssertionError(
                    f"{name}: chunk {index} emitted forbidden token(s) {sorted(hit)[:5]}"
                )


# ---------------------------------------------------------------------------
# Conditioned prefix tensors
# ---------------------------------------------------------------------------


def trace_prefix(model: Any, variant: Variant, chunk: np.ndarray) -> dict[str, Any]:
    """Capture the assembled prefix for a conditioned run, from one prefill.

    ``pre.prefix`` pins the prefix's length *and* its order (docs/TOKENIZER.md
    section 5) in a single comparison. Only the prefill is run -- the decode
    loop would add minutes and contributes nothing here.
    """
    from msl.hooks import Tracer

    lm = model._model
    conditions = model._build_conditions(
        torch.from_numpy(chunk).unsqueeze(0),
        instrument_group=instrument_group_string(variant.instruments),
    )

    tracer = Tracer()
    with tracer.attach(lm, len(lm.transformer.layers)):
        steps = lm.generate(
            prompt=None,
            conditions=conditions,
            max_gen_len=1,
            use_sampling=False,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            cfg_coef=1.0,
            early_stop_on_token=model._tokenizer.eos_id,
            beam_size=1,
            forbidden_tokens=None,
        )
        next(iter(steps))

    suffix = f".{variant.name}"
    tensors = {
        f"cond.instrument_group{suffix}": tracer.tensors["cond.instrument_group"],
        f"pre.prefix{suffix}": tracer.tensors["pre.prefix"],
    }

    prefix = tensors[f"pre.prefix{suffix}"]
    rows = tensors[f"cond.instrument_group{suffix}"]
    info = {
        "instrument_group_frames": int(rows.shape[0]),
        # mel frames + one dataset row + one row per selected instrument, then
        # the initial token.
        "prepend_length": int(prefix.shape[0]) - 1,
    }

    expected_rows = len(conditioning_rows(variant.instruments))
    if info["instrument_group_frames"] != expected_rows:
        raise AssertionError(
            f"{variant.name}: instrument_group produced "
            f"{info['instrument_group_frames']} rows, expected {expected_rows}"
        )

    return {"tensors": tensors, "info": info}
