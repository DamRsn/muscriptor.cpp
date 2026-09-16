"""Freeze the reference's integer tables so the C++ port can assert against them.

Everything here is model-independent -- no weights, no inference, no torch
tensors -- so the output is *committed* to ``testdata/vectors/`` rather than
regenerated alongside the reference dumps. That is what lets the tokenizer and
note-assembly tests run on a bare checkout.

The instrument group map is the reason this exists. Upstream builds the
singleton groups by iterating a Python ``set`` of unassigned programs::

    not_assigned = set(range(128)) - set(assigned)
    for p in not_assigned:
        ret[len(ret)] = [p]

so the group-id to program assignment for gids >= 36 is an artifact of CPython's
set iteration order, not of anything written down. The C++ side asserts against
this dump rather than reimplementing it.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from msl import paths

INSTRUMENT_VOCABULARY = "MT3_FULL_PLUS"
MAX_SHIFT_STEPS = 1001
FRAME_RATE = 100

# Instrument selections the C++ side asserts `forbidden_token_ids` against.
# Chosen to cover: the unconditional path, one melodic group, drums alone (the
# only name that is not a program group), and the two together.
FORBIDDEN_SELECTIONS: tuple[tuple[str, ...], ...] = (
    (),
    ("electric_bass",),
    ("drums",),
    ("drums", "electric_bass"),
)

# Open-note sets the C++ side asserts `tie_section_token_ids` against. Covers:
# nothing open (a bare `tie`), one note, several pitches sharing a program (the
# program token must appear once, not per pitch), two programs interleaved, and
# input given out of order (the encoder sorts).
TIE_SECTION_CASES: tuple[tuple[tuple[int, int], ...], ...] = (
    (),
    ((0, 60),),
    ((0, 60), (0, 64), (0, 67)),
    ((0, 60), (33, 40), (0, 62)),
    ((33, 40), (0, 67), (0, 60)),
)


def _vocab_ranges(vocab: list[Any]) -> dict[str, list[int]]:
    """First and last token id of each event type, in vocabulary order."""
    ranges: dict[str, list[int]] = {}
    for index, event in enumerate(vocab):
        if event.type not in ranges:
            ranges[event.type] = [index, index]
        else:
            ranges[event.type][1] = index
    return ranges


def _spot_check(vocab: list[Any], ranges: dict[str, list[int]]) -> list[list[Any]]:
    """``[id, type, value]`` at every range boundary and one id either side.

    A full round-trip test also catches an off-by-one, but only tells you the
    mapping is wrong somewhere. These name the boundary that moved.
    """
    wanted: set[int] = set()
    for first, last in ranges.values():
        wanted.update({first - 1, first, first + 1, last - 1, last, last + 1})
    return [
        [i, vocab[i].type, vocab[i].value]
        for i in sorted(wanted)
        if 0 <= i < len(vocab)
    ]


def build() -> dict[str, Any]:
    """Collect every frozen table into one JSON-serializable dict."""
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer
    from muscriptor.tokenizer.notes import DRUM_PROGRAM, SPECIAL_TOKENS

    from muscriptor.transcription_model import _build_instrument_for_program

    tokenizer = MT3Tokenizer(
        instrument_vocabulary=INSTRUMENT_VOCABULARY,
        max_shift_steps=MAX_SHIFT_STEPS,
        frame_rate=FRAME_RATE,
    )
    vocab = tokenizer._vocab
    ranges = _vocab_ranges(vocab)
    instrument_for_program = _build_instrument_for_program(tokenizer)

    # Programs 0-129 are what a `program` token can carry; DRUM_PROGRAM is
    # assigned downstream to drum hits and never decoded, but the name lookup
    # has to handle it because that is how drums reach the note list.
    program_to_name = {
        str(program): instrument_for_program(program)
        for program in list(range(130)) + [DRUM_PROGRAM]
    }

    return {
        "provenance": {
            "instrument_vocabulary": INSTRUMENT_VOCABULARY,
            "misc_programs": "SINGLETON_GROUPS",
            "is_mt3": True,
            "max_shift_steps": MAX_SHIFT_STEPS,
            "frame_rate": FRAME_RATE,
        },
        "vocab": {
            "num_tokens": tokenizer.num_tokens,
            "eos_id": tokenizer.eos_id,
            "special_tokens": list(SPECIAL_TOKENS),
            "ranges": ranges,
            "spot_check": _spot_check(vocab, ranges),
        },
        "instrument_groups": {
            "drum_program": DRUM_PROGRAM,
            "names": dict(MT3_FULL_PLUS_GROUP_NAMES),
            "group_program_map": {
                str(gid): programs
                for gid, programs in sorted(tokenizer.group_program_map.items())
            },
            "program_to_name": program_to_name,
        },
        "forbidden_token_ids": [
            {
                "instruments": list(selection),
                "token_ids": tokenizer.forbidden_token_ids(selection),
            }
            for selection in FORBIDDEN_SELECTIONS
        ],
        "tie_section_token_ids": [
            {
                "open_keys": [list(key) for key in case],
                "token_ids": tokenizer.tie_section_token_ids(case),
            }
            for case in TIE_SECTION_CASES
        ],
    }


def check(tables: dict[str, Any]) -> None:
    """Assert the dumped tables are internally consistent.

    Cheap, and it fails here rather than as a mystifying C++ mismatch.
    """
    vocab = tables["vocab"]
    ranges = vocab["ranges"]

    covered = sorted(
        index for first, last in ranges.values() for index in range(first, last + 1)
    )
    if covered != list(range(vocab["num_tokens"])):
        raise AssertionError("vocabulary ranges do not tile [0, num_tokens) exactly")

    groups = tables["instrument_groups"]
    representatives = {
        programs[0] for programs in groups["group_program_map"].values() if programs
    }
    named_representatives = {
        groups["group_program_map"][str(gid)][0]
        for gid in groups["names"].values()
        if groups["group_program_map"].get(str(gid))
    }
    if not named_representatives <= representatives:
        raise AssertionError("a named group's representative is not a representative")

    # Every program a `program` token can carry must resolve to a name, even if
    # only to "program_<n>" -- the C++ lookup has no other fallback.
    if any(not name for name in groups["program_to_name"].values()):
        raise AssertionError("program_to_name has an empty entry")

    # The gid-36 quirk, asserted rather than assumed: TOKENIZER.md section 5
    # decided to reproduce it, so a change upstream should break this, loudly.
    if groups["program_to_name"]["96"] != "drums":
        raise AssertionError(
            "program 96 no longer maps to 'drums'; the reference quirk documented "
            "in docs/TOKENIZER.md section 5 has changed and the port must follow"
        )


def dump(tables: dict[str, Any]) -> "paths.Path":
    """Write ``tables`` to ``testdata/vectors/tables.json``. Returns its path."""
    paths.VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.VECTORS_DIR / "tables.json"
    out.write_text(json.dumps(tables, indent=2, sort_keys=True) + "\n")
    return out


# ---------------------------------------------------------------------------
# C++ emission
# ---------------------------------------------------------------------------

_CPP_HEADER = """\
// GENERATED FILE -- do not edit. Regenerate with `uv run msl-tables --emit-cpp`.
//
// The MT3_FULL_PLUS instrument groups, read off the reference tokenizer rather
// than reimplemented. Upstream assigns the singleton groups (36 and up) by
// iterating a Python set of unassigned programs, so their group-id to program
// mapping is an artifact of CPython's set ordering and not of anything written
// down. test_instrument_groups.cpp asserts this table against
// testdata/vectors/tables.json, which is what keeps the two from drifting.
"""


def emit_cpp(tables: dict[str, Any]) -> str:
    """Render the frozen group table as an includable C++ fragment."""
    groups = tables["instrument_groups"]
    group_map = groups["group_program_map"]
    num_groups = len(group_map)

    representatives = [group_map[str(gid)][0] for gid in range(num_groups)]
    named = sorted(groups["names"].items(), key=lambda kv: kv[1])

    lines = [_CPP_HEADER, ""]
    lines.append(f"constexpr int NUM_GROUPS = {num_groups};")
    lines.append("")
    lines.append("// Cross-checks the value the public header hard-codes.")
    lines.append(f"constexpr int REFERENCE_DRUM_PROGRAM = {groups['drum_program']};")
    lines.append("")
    lines.append(
        "// Group id -> the group's first program, which is the only one the model"
    )
    lines.append("// ever emits for that group.")
    lines.append(
        "constexpr std::array<std::int16_t, NUM_GROUPS> GROUP_REPRESENTATIVE {{"
    )
    for start in range(0, num_groups, 12):
        row = ", ".join(f"{p:3d}" for p in representatives[start : start + 12])
        lines.append(f"    {row},")
    lines.append("}};")
    lines.append("")
    lines.append("// The user-facing names. Group ids without one surface as their")
    lines.append("// program number instead, matching the reference's program_<n>.")
    lines.append(f"constexpr std::array<NamedGroup, {len(named)}> NAMED_GROUPS {{{{")
    for name, gid in named:
        lines.append(f'    {{"{name}", {gid}}},')
    lines.append("}};")
    lines.append("")
    return "\n".join(lines)


def emit_cpp_file(tables: dict[str, Any]) -> "paths.Path":
    """Write ``cpp/src/instrument_groups.inc``. Returns its path."""
    out = paths.CPP_DIR / "src" / "instrument_groups.inc"
    out.write_text(emit_cpp(tables))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-cpp",
        action="store_true",
        help="also regenerate cpp/src/instrument_groups.inc",
    )
    args = parser.parse_args()

    tables = build()
    check(tables)
    out = dump(tables)
    print(f"wrote {out}")
    print(f"  {tables['vocab']['num_tokens']} tokens, "
          f"{len(tables['instrument_groups']['group_program_map'])} instrument groups, "
          f"{len(tables['instrument_groups']['names'])} named")

    if args.emit_cpp:
        print(f"wrote {emit_cpp_file(tables)}")
