"""Drive the reference's token-to-note path and record it at every level.

Shared by the two fixture generators -- ``note_vectors`` (hand-authored token
sequences, no model) and ``note_refs`` (real model output). They must decode
identically or the C++ side is being held to two different standards, so the
decoding lives here once rather than in both.

Three levels are recorded for every stream, because they are three separate
ports and a wrong note is otherwise a mystery split between them:

``actions``    what ``OpenNoteTracker`` emits. The decode state machine alone.
``events``     the public NoteStart/NoteEnd stream. Adds instrument naming.
``notes``      reassembled, validated and overlap-trimmed. Adds the cleanup.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Iterator

from msl import tables


def build_tokenizer() -> Any:
    from muscriptor.tokenizer.mt3 import MT3Tokenizer

    return MT3Tokenizer(
        instrument_vocabulary=tables.INSTRUMENT_VOCABULARY,
        max_shift_steps=tables.MAX_SHIFT_STEPS,
        frame_rate=tables.FRAME_RATE,
    )


def name_to_program_map(tokenizer: Any) -> dict[str, int]:
    """Instrument name to its group's representative program.

    The inverse of the decode-side lookup, and the same one
    ``TranscriptionModel._program_for_instrument`` builds.
    """
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES

    return {
        name: tokenizer.group_program_map[gid][0]
        for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items()
        if tokenizer.group_program_map.get(gid)
    }


def describe_action(action: Any) -> dict[str, Any]:
    from muscriptor.events import _DrumHit, _EndNote, _StartNote

    if isinstance(action, _StartNote):
        return {
            "kind": "start",
            "program": action.program,
            "pitch": action.pitch,
            "time": action.time,
        }

    if isinstance(action, _EndNote):
        return {
            "kind": "end",
            "program": action.program,
            "pitch": action.pitch,
            "time": action.time,
        }

    if isinstance(action, _DrumHit):
        return {"kind": "drum", "pitch": action.pitch, "time": action.time}

    raise TypeError(f"unexpected note action {action!r}")


def describe_event(event: Any) -> dict[str, Any]:
    from muscriptor.events import NoteStartEvent

    if isinstance(event, NoteStartEvent):
        return {
            "kind": "start",
            "index": event.index,
            "pitch": event.pitch,
            "time": event.start_time,
            "instrument": event.instrument,
        }

    return {
        "kind": "end",
        "start_index": event.start_event_index,
        "time": event.end_time,
    }


def note_json(note: Any) -> dict[str, Any]:
    return {
        "is_drum": bool(note.is_drum),
        "program": int(note.program),
        "pitch": int(note.pitch),
        "onset": float(note.onset),
        "offset": float(note.offset),
    }


def assemble_notes(events: Iterable[Any], name_to_program: dict[str, int]) -> list[Any]:
    """Reassemble Notes from a NoteStart/NoteEnd stream.

    Mirrors ``TranscriptionModel.events_to_midi_bytes`` up to the point where it
    starts writing MIDI -- which is exactly where this library stops too. Notes
    are appended in *close* order, and that order is load-bearing: the stable
    sort in ``trim_overlapping_notes`` inherits it.
    """
    from muscriptor.events import NoteStartEvent
    from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note

    notes: list[Any] = []
    open_notes: dict[int, Any] = {}

    for event in events:
        if isinstance(event, NoteStartEvent):
            is_drum = event.instrument == "drums"

            if is_drum:
                program = DRUM_PROGRAM
            elif event.instrument in name_to_program:
                program = name_to_program[event.instrument]
            elif event.instrument.startswith("program_"):
                program = int(event.instrument.removeprefix("program_"))
            else:
                raise ValueError(f"unknown instrument name {event.instrument!r}")

            open_notes[event.index] = Note(
                is_drum=is_drum,
                program=program,
                onset=event.start_time,
                offset=event.start_time,
                pitch=event.pitch,
            )
        else:
            note = open_notes.pop(event.start_event_index)
            note.offset = event.end_time
            notes.append(note)

    if open_notes:
        raise AssertionError(f"{len(open_notes)} note starts never closed")

    return notes


def clean_notes(notes: list[Any]) -> list[Any]:
    """``validate_notes`` then ``trim_overlapping_notes``, on a copy.

    Exactly the two passes ``events_to_midi_bytes`` runs, and only those two.
    ``note_event2note``'s ten-second runaway guard is *not* on the inference
    path -- it is reachable only from upstream's own test suite -- so applying
    it here would silently truncate legitimate long notes.
    """
    from muscriptor.tokenizer.notes import trim_overlapping_notes, validate_notes

    return trim_overlapping_notes(validate_notes(copy.deepcopy(notes), fix=True))


def decode(
    stream_items: Iterator[Any],
    tokenizer: Any,
    instrument_for_program: Any,
) -> dict[str, Any]:
    """Run one interleaved boundary/token stream through the reference.

    @param stream_items Interleaved ``ChunkBoundary`` markers and token ids,
           in the order the decoder consumes them.
    @return The three recorded levels, plus the raw pre-cleanup note list.
    """
    from muscriptor.events import decode_model_tokens

    items = list(stream_items)
    events = list(
        decode_model_tokens(
            iter(items), tokenizer._vocab, instrument_for_program, tokenizer.frame_rate
        )
    )
    raw = assemble_notes(events, name_to_program_map(tokenizer))

    return {
        "actions": run_tracker(items, tokenizer)["actions"],
        "events": [describe_event(e) for e in events],
        "notes_raw": [note_json(n) for n in raw],
        "notes": [note_json(n) for n in clean_notes(raw)],
    }


def run_tracker(items: list[Any], tokenizer: Any) -> dict[str, Any]:
    """Replay `items` through a bare ``OpenNoteTracker``.

    Separate from :func:`decode` because the actions are the bisection point
    between the state machine and the assembler, and because ``open_keys()`` at
    each boundary is what prelude forcing reads -- the C++ side can check both
    without ever running the transformer.
    """
    from muscriptor.events import ChunkBoundary, OpenNoteTracker

    tracker = OpenNoteTracker(tokenizer._vocab, tokenizer.frame_rate)
    actions: list[dict[str, Any]] = []
    open_keys: list[list[list[int]]] = []

    for item in items:
        actions.extend(describe_action(a) for a in tracker.feed(item))

        # Read after feeding the boundary, never before: the boundary settles a
        # previous chunk that never emitted its tie token, and only then is
        # open_keys() the decoder's own view. Getting this order wrong is what
        # makes a forced prologue disagree with the tracker it came from.
        if isinstance(item, ChunkBoundary):
            open_keys.append([list(key) for key in tracker.open_keys()])

    actions.extend(describe_action(a) for a in tracker.finish())
    return {"actions": actions, "open_keys": open_keys}


def check(result: dict[str, Any], label: str) -> None:
    """Assert the recorded levels agree with each other."""
    starts = sum(1 for a in result["actions"] if a["kind"] == "start")
    ends = sum(1 for a in result["actions"] if a["kind"] == "end")
    drums = sum(1 for a in result["actions"] if a["kind"] == "drum")

    if starts != ends:
        raise AssertionError(f"{label}: {starts} note starts but {ends} ends")

    if len(result["notes_raw"]) != starts + drums:
        raise AssertionError(
            f"{label}: {len(result['notes_raw'])} raw notes from "
            f"{starts + drums} start/drum actions"
        )

    for note in result["notes"]:
        if note["offset"] <= note["onset"]:
            raise AssertionError(f"{label}: a cleaned note has non-positive duration")
