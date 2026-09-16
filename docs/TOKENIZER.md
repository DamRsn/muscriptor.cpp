# MT3 tokenizer

MuScriptor's "tokenizer" is not a text tokenizer. It has no BPE and no
vocabulary file. It is a fixed mapping between 1393 token ids and events, plus
a state machine that turns those events into notes. It is integer logic with no
learned parameters.

The design comes from MT3 via YourMT3+. MuScriptor uses the `MT3_FULL_PLUS`
variant, with `max_shift_steps = 1001` and `frame_rate = 100`. Everything below
follows `tokenizer/{mt3,notes}.py`, `events.py` and `transcription_model.py`
in the [reference implementation](https://github.com/muscriptor/muscriptor).

## 1. Vocabulary

`build_event_vocab(max_shift_steps=1001)` joins contiguous ranges in a fixed
order; a token's id is its position in that list.

| Token ids | Type | Values | Meaning |
|---|---|---|---|
| 0 | `PAD` | — | never produced during inference |
| 1 | `EOS` | — | end of chunk |
| 2 | `UNK` | — | never produced during inference |
| 3 – 1003 | `shift` | 0 – 1000 | time within the chunk, in 10 ms steps from its start |
| 1004 – 1131 | `pitch` | 0 – 127 | MIDI note number |
| 1132 – 1133 | `velocity` | 0 – 1 | note on/off flag, not dynamics |
| 1134 | `tie` | 0 | end of the tie prologue |
| 1135 – 1264 | `program` | 0 – 129 | MIDI program |
| 1265 – 1392 | `drum` | 0 – 127 | GM percussion note, instantaneous hit |

That makes 1393 tokens. `small` has `card = 1393`; `medium` and `large` have
`card = 1395`. `LMModel._compute_logits` always sets `logits[:, 1393:]` to
−inf, so ids 1393 and 1394 are never produced (`Hparams::logit_mask_start`).

`initial_token_id = card` is the BOS token fed at prefill. The embedding table
has `card + 1` rows.

`velocity` is binary: 1 turns a note on, 0 turns it off. The reference writes
every MIDI note at velocity 100.

## 2. Chunk grammar

Each 5 s chunk produces one token sequence:

```
chunk    ::= prologue body EOS
prologue ::= ( program pitch+ )* tie
body     ::= ( shift | program | velocity | pitch | drum )*
```

The prologue (tie section) lists the `(program, pitch)` notes still sounding
from the previous chunk. The body gives onsets and offsets.

`program` and `velocity` are sticky: each sets a register that holds until the
next token of that type. `pitch` acts on the current `(program, velocity)`.

## 3. Decode state machine (`OpenNoteTracker`)

The following reset at every chunk boundary: `tick_state`, `program`,
`velocity`, `in_prologue = true`, `skip_rest = false`, `tie_set = {}`. The
open-note map `(program, pitch) → onset` carries over between chunks.

`start_tick = round(seek_time * 100)`, and `tick_state` starts there.

**In the prologue:**

- `program` sets the program register.
- `pitch`, once a program is set, adds `(program, pitch)` to `tie_set`.
- `tie` ends the prologue. Every open note not in `tie_set` closes at
  `seek_time`, and the velocity register is cleared.
- `shift` means the chunk is malformed. Every open note closes at `seek_time`,
  `skip_rest` is set, and the rest of the chunk is ignored.

**In the body:**

- `shift v`: if `v > 0`, `tick_state = start_tick + v`. `shift 0` does nothing.
- `program v` / `velocity v` set the matching register.
- `drum v` emits a hit with pitch `v` at `tick_state / 100`. Hits never enter
  the open-note map, and the velocity register does not affect them.
- `pitch v` is ignored unless both registers are set. Otherwise, with
  `t = tick_state / 100` and `key = (program, v)`:
  - if `key` is open, it closes at `t`;
  - then, if `velocity > 0`, `key` opens at `t`.

  An already-open pitch that arrives with velocity 1 therefore retriggers: it
  closes and reopens at the same time.

**Window rule:** in the body, a `pitch` or `drum` event with
`t >= next_seek_time` is dropped.

**End of stream (`finish`):** if the last chunk ended inside its prologue,
every open note closes at that chunk's `seek_time`. Otherwise each open note
closes at `onset + 0.01 s`.

## 4. Prelude forcing

Prelude forcing is on by default. At each chunk boundary after the first, the
notes still open are encoded and passed as a forced prompt, instead of letting
the model predict its own prologue:

```
for (program, pitch) in sorted(open_keys):     # by (program, pitch)
    if program != last_program: emit program; last_program = program
    emit pitch
emit tie
```

The prompt is prefilled together with the conditioning. The forced tokens also
go through the tracker, like generated ones. Chunks have to be decoded in order
(`batch_size = 1`).

The `tokens` block of the reference dump is decoded with prelude forcing off,
so every chunk can be checked on its own. The `prelude` variant in `notes.json`
is decoded with it on ([`TESTING.md`](TESTING.md)).

## 5. Instrument groups

The model decodes a `program` from 130 values, but only ever emits the first
program of a group, its representative.

`MT3_FULL_PLUS` defines groups 0–35, covering GM programs 0–95, 100 and 101.
`SINGLETON_GROUPS` then gives each remaining program ({96–99} ∪ {102–127}) a
group of its own, for 66 groups in total. 34 groups have names (0–33), and
`drums` is group 36:

| Name | gid | prog | Name | gid | prog |
|---|---|---|---|---|---|
| acoustic_piano | 0 | 0 | trumpet | 19 | 56 |
| electric_piano | 1 | 2 | trombone | 20 | 57 |
| chromatic_percussion | 2 | 8 | tuba | 21 | 58 |
| organ | 3 | 16 | french_horn | 22 | 60 |
| acoustic_guitar | 4 | 24 | brass_section | 23 | 61 |
| clean_electric_guitar | 5 | 26 | soprano_and_alto_sax | 24 | 64 |
| distorted_electric_guitar | 6 | 29 | tenor_sax | 25 | 66 |
| acoustic_bass | 7 | 32 | baritone_sax | 26 | 67 |
| electric_bass | 8 | 33 | oboe | 27 | 68 |
| violin | 9 | 40 | english_horn | 28 | 69 |
| viola | 10 | 41 | bassoon | 29 | 70 |
| cello | 11 | 42 | clarinet | 30 | 71 |
| contrabass | 12 | 43 | flutes | 31 | 72 |
| orchestral_harp | 13 | 46 | synth_lead | 32 | 80 |
| timpani | 14 | 47 | synth_pad | 33 | 88 |
| string_ensemble | 15 | 48 | drums | 36 | (96) |
| synth_strings | 16 | 50 | | | |
| voice | 17 | 52 | | | |
| orchestra_hit | 18 | 55 | | | |

Groups 34 and 35 and the singleton groups have no name. A program in one of
them is labelled `program_<n>`.

**The group table is generated.** `get_group_program_map` builds the singleton
groups by iterating a Python `set`, so which program lands in which group id
above 35 depends on CPython's set ordering. `uv run msl-tables --emit-cpp`
writes the table to `cpp/src/instrument_groups.inc`, and
`test_instrument_groups.cpp` checks that file against the same dump.

**Drums are not a program.** They arrive as `drum` tokens and get
`DRUM_PROGRAM = 128`. Group 36 is how drums are selected for conditioning and
filtering.

**Program 96 is labelled drums.** Group 36 is also the singleton group of GM
program 96. The reference maps a decoded program 96 to `"drums"` and writes it
to MIDI channel 10. The port does the same: `instrumentGroupFor(96)` returns
`Drums`. The forbidden-token mask handles `"drums"` separately and is not
affected.

### Restricting instruments

Selecting instruments applies two separate mechanisms, as the reference does:

1. **Conditioning (soft).** The selected group ids are embedded and inserted
   into the prefix, one position per group. With no selection there is exactly
   one position, the null class at embedding row 1. Group id `g` uses row
   `g + 2`. The prefix is
   `[mel, dataset_name, instrument_group…, tokens]`
   (`Model::setInstrumentRows`).
2. **Forbidden tokens (hard).** Every `program` token that is not a selected
   group's representative is set to −inf before the argmax, and so is every
   `drum` token unless `drums` is selected. Timing, pitch, velocity, tie and
   special tokens are never masked (`InstrumentGroups::forbiddenTokenIds`,
   `Model::setForbiddenTokens`).

`TranscribeOptions::instruments` turns on both ([`API.md`](API.md)).

## 6. Note assembly

Note starts and ends are assembled into `Note{is_drum, program, onset, offset,
pitch}`, followed by two cleanup passes:

- **`validate_notes(fix=True)`** is an if/else-if chain, evaluated in this
  order:

  ```
  onset missing                            → drop the note
  else if offset missing                   → offset = onset + 0.01
  else if onset > offset                   → offset = max(offset, onset + 0.01)
  else if !is_drum and shorter than 0.01 s → offset = onset + 0.01
  ```

  Only the last two branches can be reached from decoding. Drums skip the
  minimum-duration branch.
- **`trim_overlapping_notes`** works within each `(program, pitch, is_drum)`
  channel. Notes are stable-sorted by onset alone, each offset is cut back to
  the next note's onset, and notes left empty are dropped. The result is then
  sorted by `(onset, is_drum, program, pitch, offset)`.

Nothing else runs. `note_event2note`'s ten-second limit on note length is not
on the inference path.

The library returns these notes. The reference goes on to write MIDI: one track
per instrument, drums on channel 10, velocity 100, 120 BPM.

## 7. Easy to get wrong

1. `shift` is a time within the chunk, not a delta, and `shift 0` does nothing.
2. The same `pitch` token declares a sustained note in the prologue and
   triggers one in the body.
3. Without the `>= next_seek_time` drop, notes at chunk boundaries are
   duplicated.
4. A retrigger closes and reopens the note at the same time, leaving a
   zero-length note that `validate_notes` widens to 10 ms.
5. Open notes carry across chunks; the registers do not.
6. Without the malformed-chunk rule (`shift` before `tie`), notes stay open for
   good.
7. The open-note map keeps insertion order, and `finish` closes notes in that
   order. A hash map changes the order of output events.

`test_tracker.cpp` and `test_note_assembly.cpp` check this logic with no
weights. `test_tokens_to_notes.cpp` replays the reference's token streams
through it.

## 8. Coverage from real output

The fixture's token streams never reach three of these rules: a malformed
chunk, the `>= next_seek_time` drop (the largest shift in each chunk is 491, 492
and 490, all under the 500-step window), and `shift 0` or a shift that goes
backwards. Hand-written vectors in `testdata/vectors/note_vectors.json` cover
them, along with duplicate drum hits on one tick. Those are the only inputs that
make `trim_overlapping_notes` change anything.
