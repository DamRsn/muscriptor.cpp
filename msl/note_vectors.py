"""Hand-authored token sequences and the notes the reference decodes them into.

The 3-chunk audio fixture is real model output, which makes it the right
end-to-end target -- and a poor unit test. Measured over its token streams, it
never once emits a `shift` before its `tie` token, never emits `shift 0`, never
emits a non-monotonic shift, and never emits an event past its own 5-second
window (its largest shifts are 491, 492 and 490 against a 500-tick window). So
four of the decode state machine's rules (`docs/TOKENIZER.md` section 8) are
not exercised by it at all.

These vectors cover them. Each is authored as a list of *events*, encoded to
token ids through the reference tokenizer's own index (never by hand-written
integers), then run through the reference's real `OpenNoteTracker`,
`decode_model_tokens`, and note-assembly and cleanup passes. The result is
committed, so the C++ tracker can be tested against it with no weights, no
reference dump and no inference.

The output is deliberately dumped at three levels -- actions, events, notes --
because they are three separate ports. A note that comes out wrong is otherwise
a mystery split between the state machine and the assembler.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from msl import note_decode, paths, tables

# An event as authored: (type, value). `tie` and the specials carry value 0.
Event = tuple[str, int]


@dataclass(frozen=True)
class Vector:
    """One token-stream test case.

    `seek_times` has one entry per chunk; `next_seek_time` for chunk i is
    `seek_times[i + 1]`, or None for the last -- which is what switches the
    window-drop rule off on the final chunk.
    """

    name: str
    description: str
    seek_times: tuple[float, ...]
    chunks: tuple[tuple[Event, ...], ...]

    def __post_init__(self) -> None:
        if len(self.seek_times) != len(self.chunks):
            raise ValueError(f"{self.name}: seek_times and chunks disagree in length")


# `tie` with nothing before it: the prologue declares no sustained notes, so
# every note open at the boundary closes there.
TIE: tuple[Event, ...] = (("tie", 0),)


def _sustain(*keys: tuple[int, int]) -> tuple[Event, ...]:
    """A tie prologue declaring `keys` sustained, in tie_section_token_ids order."""
    events: list[Event] = []
    program_state: int | None = None
    for program, pitch in sorted(keys):
        if program != program_state:
            events.append(("program", program))
            program_state = program
        events.append(("pitch", pitch))
    events.append(("tie", 0))
    return tuple(events)


VECTORS: tuple[Vector, ...] = (
    # --- rules the audio fixture never reaches -----------------------------
    Vector(
        name="shift_zero_is_noop",
        description=(
            "`shift 0` does not rewind to the chunk start. Treating shift as a "
            "delta rather than an absolute offset would put pitch 62 at 0.00 "
            "instead of 0.50, which looks entirely plausible in a piano roll."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 50),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 0),
                ("velocity", 1),
                ("pitch", 62),
                ("shift", 80),
                ("velocity", 0),
                ("pitch", 60),
                ("velocity", 0),
                ("pitch", 62),
            ),
        ),
    ),
    Vector(
        name="shift_is_absolute_within_chunk",
        description=(
            "Two shifts in one chunk land at start_tick + value each, not "
            "cumulatively. In chunk 1 (seek 5.0) shift 30 then shift 50 means "
            "5.30 then 5.50, never 5.80."
        ),
        seek_times=(0.0, 5.0),
        chunks=(
            TIE + (("shift", 10), ("program", 0), ("velocity", 1), ("pitch", 60)),
            _sustain((0, 60))
            + (
                ("shift", 30),
                ("velocity", 0),
                ("pitch", 60),
                ("shift", 50),
                ("velocity", 1),
                ("pitch", 62),
                ("shift", 90),
                ("velocity", 0),
                ("pitch", 62),
            ),
        ),
    ),
    Vector(
        name="window_drop_and_last_chunk_exemption",
        description=(
            "Chunk 0 emits a pitch and a drum past its own window; both are "
            "dropped because their time is >= next_seek_time. The final chunk "
            "emits the identical shifts and keeps both, because next_seek_time "
            "is None there. Skipping this rule duplicates notes at every seam."
        ),
        seek_times=(0.0, 5.0),
        chunks=(
            TIE
            + (
                ("shift", 490),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 495),
                ("velocity", 0),
                ("pitch", 60),
                # Past the window: both dropped.
                ("shift", 510),
                ("velocity", 1),
                ("pitch", 62),
                ("shift", 600),
                ("drum", 36),
            ),
            TIE
            + (
                ("shift", 510),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 62),
                ("shift", 600),
                ("drum", 36),
                ("shift", 650),
                ("velocity", 0),
                ("pitch", 62),
            ),
        ),
    ),
    Vector(
        name="malformed_chunk_closes_all_and_skips_rest",
        description=(
            "A `shift` arriving before the `tie` token means the chunk never "
            "wrote a prologue: every open note closes at the boundary and the "
            "rest of that chunk is discarded, including a `tie` that shows up "
            "later. Chunk 2 then decodes normally. Ignoring this rule leaves "
            "notes open forever."
        ),
        seek_times=(0.0, 5.0, 10.0),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("pitch", 64),
            ),
            (
                ("shift", 20),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 67),
                # Discarded: skip_rest is already set.
                ("tie", 0),
                ("shift", 30),
                ("pitch", 72),
            ),
            TIE
            + (
                ("shift", 40),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 55),
                ("shift", 90),
                ("velocity", 0),
                ("pitch", 55),
            ),
        ),
    ),
    Vector(
        name="eos_in_prologue_closes_all",
        description=(
            "The last chunk's tokens run out mid-prologue, before its `tie`. "
            "finish() then closes everything at that chunk's seek_time rather "
            "than at onset + the minimum duration."
        ),
        seek_times=(0.0, 5.0),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("pitch", 64),
            ),
            (("program", 0), ("pitch", 60)),
        ),
    ),
    # --- rules the fixture does reach, kept because they localise failures ---
    Vector(
        name="tie_prologue_partial",
        description=(
            "Three notes cross a boundary; the prologue re-declares one. The "
            "other two close at the boundary time, the declared one keeps its "
            "original onset."
        ),
        seek_times=(0.0, 5.0),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("pitch", 64),
                ("pitch", 67),
            ),
            _sustain((0, 64))
            + (("shift", 20), ("velocity", 0), ("pitch", 64)),
        ),
    ),
    Vector(
        name="tie_prologue_two_programs",
        description=(
            "The prologue's `program` token is sticky: it is emitted once per "
            "run of pitches, not once per pitch. This is also the exact layout "
            "prelude forcing has to reproduce."
        ),
        seek_times=(0.0, 5.0),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("pitch", 64),
                ("program", 33),
                ("velocity", 1),
                ("pitch", 40),
            ),
            _sustain((0, 60), (0, 64), (33, 40))
            + (
                ("shift", 20),
                ("program", 0),
                ("velocity", 0),
                ("pitch", 60),
                ("pitch", 64),
                ("program", 33),
                ("pitch", 40),
            ),
        ),
    ),
    Vector(
        name="retrigger",
        description=(
            "A pitch already open that gets velocity 1 again closes and "
            "reopens at the same instant."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 20),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 40),
                ("velocity", 0),
                ("pitch", 60),
            ),
        ),
    ),
    Vector(
        name="retrigger_zero_length_is_dropped",
        description=(
            "Retriggering at the same tick produces a zero-length note. "
            "validate_notes widens it to the 10 ms minimum, then "
            "trim_overlapping_notes clamps it back against the next onset and "
            "drops it. The action list still contains it; the note list must "
            "not."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 40),
                ("velocity", 0),
                ("pitch", 60),
            ),
        ),
    ),
    Vector(
        name="pitch_without_registers_is_ignored",
        description=(
            "In the body a `pitch` needs both a program and a velocity "
            "register set; without either it is silently dropped rather than "
            "defaulted."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("pitch", 60),
                ("program", 0),
                ("pitch", 62),
                ("velocity", 1),
                ("pitch", 64),
                ("shift", 30),
                ("velocity", 0),
                ("pitch", 64),
            ),
        ),
    ),
    Vector(
        name="drum_ignores_velocity_and_program",
        description=(
            "Drum hits are instantaneous and read neither register: they fire "
            "with velocity 0 set and with no program ever declared."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("velocity", 0),
                ("drum", 36),
                ("shift", 20),
                ("drum", 38),
            ),
        ),
    ),
    Vector(
        name="drum_duplicate_at_same_tick",
        description=(
            "Two identical drum hits at one tick both reach the action list, "
            "then trim_overlapping_notes collapses them to one. This is the "
            "only path on monotonic output where trimming changes anything."
        ),
        seek_times=(0.0,),
        chunks=(TIE + (("shift", 10), ("drum", 36), ("drum", 36)),),
    ),
    Vector(
        name="program_96_decodes_as_drums",
        description=(
            "The deliberate reference quirk from docs/TOKENIZER.md section 5: "
            "group 36 is the singleton group for GM program 96, so a decoded "
            "program 96 is named 'drums' and routed as a drum. Reproduced on "
            "purpose; this vector is what makes changing that a visible "
            "decision rather than a silent divergence."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 96),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 30),
                ("velocity", 0),
                ("pitch", 60),
            ),
        ),
    ),
    Vector(
        name="finish_minimum_duration",
        description=(
            "Notes still open when the stream ends close at onset + 10 ms, "
            "and they do so in insertion order."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 20),
                ("pitch", 64),
            ),
        ),
    ),
    Vector(
        name="insertion_order_survives_to_finish",
        description=(
            "Three notes opened in an order that differs from (program, pitch) "
            "order. finish() replays their closes in insertion order, so the "
            "action list pins it. Storing the open set in a std::map or an "
            "unordered_map reorders this and nothing else notices."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 10),
                ("program", 33),
                ("velocity", 1),
                ("pitch", 40),
                ("shift", 20),
                ("program", 0),
                ("pitch", 60),
                ("shift", 30),
                ("pitch", 64),
            ),
        ),
    ),
    Vector(
        name="trim_out_of_order_shifts",
        description=(
            "Shift is absolute and nothing at inference enforces that it "
            "increases, so a chunk can emit an offset earlier than the onset "
            "it closes. validate_notes repairs the inverted note and "
            "trim_overlapping_notes then truncates the enclosing one. The "
            "audio fixture never does this -- its shifts are strictly "
            "monotonic -- which is exactly why it is here."
        ),
        seek_times=(0.0,),
        chunks=(
            TIE
            + (
                ("shift", 300),
                ("program", 0),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 100),
                ("velocity", 0),
                ("pitch", 60),
                ("shift", 200),
                ("velocity", 1),
                ("pitch", 60),
                ("shift", 400),
                ("velocity", 0),
                ("pitch", 60),
            ),
        ),
    ),
    Vector(
        name="empty_chunk",
        description="A chunk that is nothing but its tie token decodes to nothing.",
        seek_times=(0.0, 5.0),
        chunks=(
            TIE + (("shift", 10), ("program", 0), ("velocity", 1), ("pitch", 60)),
            TIE,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Running a vector through the reference
# ---------------------------------------------------------------------------


def _encode(tokenizer: Any, events: tuple[Event, ...]) -> list[int]:
    """Events to token ids through the tokenizer's own index."""
    return [tokenizer._token_index[(etype, value)] for etype, value in events]


def run(vector: Vector) -> dict[str, Any]:
    """Decode one vector through the reference, at all three levels."""
    from muscriptor.events import ChunkBoundary
    from muscriptor.transcription_model import _build_instrument_for_program

    tokenizer = note_decode.build_tokenizer()
    chunk_tokens = [_encode(tokenizer, chunk) for chunk in vector.chunks]

    items: list[Any] = []
    for index, (seek, tokens) in enumerate(zip(vector.seek_times, chunk_tokens, strict=True)):
        next_seek = (
            vector.seek_times[index + 1] if index + 1 < len(vector.seek_times) else None
        )
        items.append(ChunkBoundary(seek_time=seek, next_seek_time=next_seek))
        items.extend(tokens)

    decoded = note_decode.decode(
        iter(items), tokenizer, _build_instrument_for_program(tokenizer)
    )
    replay = note_decode.run_tracker(items, tokenizer)

    return {
        "name": vector.name,
        "description": vector.description,
        "seek_times": list(vector.seek_times),
        "chunk_events": [
            [[etype, value] for etype, value in chunk] for chunk in vector.chunks
        ],
        "chunk_tokens": chunk_tokens,
        "open_keys_at_boundary": replay["open_keys"],
        **decoded,
    }


def build() -> dict[str, Any]:
    results = [run(vector) for vector in VECTORS]
    for result in results:
        note_decode.check(result, result["name"])
    return {
        "provenance": {
            "instrument_vocabulary": tables.INSTRUMENT_VOCABULARY,
            "max_shift_steps": tables.MAX_SHIFT_STEPS,
            "frame_rate": tables.FRAME_RATE,
            "minimum_note_duration_sec": 0.01,
        },
        "vectors": results,
    }


def dump(payload: dict[str, Any]) -> "paths.Path":
    """Write ``payload`` to ``testdata/vectors/note_vectors.json``. Returns its path."""
    paths.VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.VECTORS_DIR / "note_vectors.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="show each vector's notes")
    args = parser.parse_args()

    payload = build()
    out = dump(payload)
    print(f"wrote {out} ({len(payload['vectors'])} vectors)")

    if args.print:
        for result in payload["vectors"]:
            print(f"\n{result['name']}")
            for note in result["notes"]:
                kind = "drum" if note["is_drum"] else f"prog {note['program']:3d}"
                print(
                    f"  {kind}  pitch {note['pitch']:3d}  "
                    f"{note['onset']:.2f} -> {note['offset']:.2f}"
                )
