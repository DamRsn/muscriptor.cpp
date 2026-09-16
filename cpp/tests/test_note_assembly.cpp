// The two cleanup passes, and the streaming slice built on top of them.
//
// validate_notes and trim_overlapping_notes look trivial and are not: the
// first is an if/else-if chain whose branches are not commutative, and the
// second depends on a *stable* sort inheriting note close order. Both are
// exercised through the vectors as well, but a failure there says "a note is
// wrong"; these say which rule.
//
// Runs with no weights and no reference dump -- tagged [pure].

#include "vectors.hpp"

#include "note_assembler.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>

using namespace msl;
using namespace msl::test;

namespace
{

Note makeNote(double inOnset, double inOffset, int inPitch = 60, int inProgram = 0, bool inIsDrum = false)
{
    Note note;
    note.onset = inOnset;
    note.offset = inOffset;
    note.pitch = inPitch;
    note.program = inProgram;
    note.is_drum = inIsDrum;
    return note;
}

} // namespace

TEST_CASE("validateNotes repairs an inverted note", "[pure][notes]")
{
    std::vector<Note> notes {makeNote(3.0, 1.0)};
    validateNotes(notes);

    REQUIRE(notes.size() == 1);
    CHECK_THAT(notes[0].offset, Catch::Matchers::WithinAbs(3.0 + MINIMUM_NOTE_DURATION_SECONDS, 1e-12));
}

TEST_CASE("validateNotes widens a note shorter than the minimum", "[pure][notes]")
{
    std::vector<Note> notes {makeNote(1.0, 1.0)};
    validateNotes(notes);

    CHECK_THAT(notes[0].offset, Catch::Matchers::WithinAbs(1.0 + MINIMUM_NOTE_DURATION_SECONDS, 1e-12));
}

TEST_CASE("validateNotes leaves drum hits at their own duration", "[pure][notes]")
{
    // The minimum-duration branch is guarded on !is_drum, and a drum hit is
    // already exactly the minimum long -- widening it again would push every
    // hit 10 ms past where the reference puts it.
    std::vector<Note> notes {makeNote(1.0, 1.0 + MINIMUM_NOTE_DURATION_SECONDS, 36, DRUM_PROGRAM, true)};
    validateNotes(notes);

    CHECK_THAT(notes[0].offset, Catch::Matchers::WithinAbs(1.0 + MINIMUM_NOTE_DURATION_SECONDS, 1e-12));
}

TEST_CASE("trimOverlappingNotes truncates against the next onset", "[pure][notes]")
{
    const std::array<Note, 2> notes {makeNote(2.0, 4.0), makeNote(3.0, 3.5)};
    const std::vector<Note> trimmed = trimOverlappingNotes(notes);

    INFO(describeNotes(trimmed));
    REQUIRE(trimmed.size() == 2);
    CHECK_THAT(trimmed[0].onset, Catch::Matchers::WithinAbs(2.0, 1e-12));
    CHECK_THAT(trimmed[0].offset, Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("trimOverlappingNotes drops a note left empty", "[pure][notes]")
{
    // Two identical drum hits on one tick: the first is truncated to the
    // second's onset, leaving it zero-length, and dropped. This is the only
    // path on monotonic model output where trimming changes anything.
    const std::array<Note, 2> notes {makeNote(1.0, 1.01, 36, DRUM_PROGRAM, true),
                                     makeNote(1.0, 1.01, 36, DRUM_PROGRAM, true)};

    const std::vector<Note> trimmed = trimOverlappingNotes(notes);
    CHECK(trimmed.size() == 1);
}

TEST_CASE("trimming is per channel, not across the whole list", "[pure][notes]")
{
    // Same pitch and onset, different programs: two instruments playing in
    // unison must not truncate each other.
    const std::array<Note, 2> notes {makeNote(1.0, 3.0, 60, 0), makeNote(2.0, 3.0, 60, 33)};
    const std::vector<Note> trimmed = trimOverlappingNotes(notes);

    INFO(describeNotes(trimmed));
    REQUIRE(trimmed.size() == 2);
    CHECK_THAT(trimmed[0].offset, Catch::Matchers::WithinAbs(3.0, 1e-12));
    CHECK_THAT(trimmed[1].offset, Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("a drum and a melodic note never share a channel", "[pure][notes]")
{
    // is_drum is part of the trimming key rather than implied by the program,
    // because a melodic note can legitimately carry program 128 too.
    const std::array<Note, 2> notes {makeNote(1.0, 3.0, 36, DRUM_PROGRAM, false),
                                     makeNote(2.0, 2.01, 36, DRUM_PROGRAM, true)};

    const std::vector<Note> trimmed = trimOverlappingNotes(notes);
    REQUIRE(trimmed.size() == 2);
    CHECK_THAT(trimmed[0].offset, Catch::Matchers::WithinAbs(3.0, 1e-12));
}

TEST_CASE("the final sort uses the reference's five keys", "[pure][notes]")
{
    std::vector<Note> notes {
        makeNote(1.0, 2.0, 64, 0),
        makeNote(1.0, 2.0, 60, 0),
        makeNote(1.0, 1.01, 38, DRUM_PROGRAM, true),
        makeNote(0.5, 2.0, 72, 0),
    };

    sortNotes(notes);

    INFO(describeNotes(notes));
    CHECK_THAT(notes[0].onset, Catch::Matchers::WithinAbs(0.5, 1e-12));
    // Same onset: melodic before drum, then by program, then by pitch.
    CHECK(notes[1].pitch == 60);
    CHECK(notes[2].pitch == 64);
    CHECK(notes[3].is_drum);
}

TEST_CASE("the streaming slice agrees with the final list", "[pure][notes]")
{
    // The callback contract: concatenating what each chunk reports has to equal
    // finalize(), or a progressively-filling piano roll ends up showing notes
    // that the finished transcription disagrees with.
    for (const NoteVector& vector: Vectors::get().noteVectors()) {
        INFO(vector.name);

        OpenNoteTracker tracker;
        NoteAssembler assembler;
        const std::vector<ChunkBoundary> boundaries = vector.boundaries();
        const int last_chunk = static_cast<int>(boundaries.size()) - 1;

        std::vector<Note> streamed;

        for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
            const int index = static_cast<int>(chunk);
            assembler.apply(tracker.feed(boundaries[chunk]), index);

            for (const std::int32_t token: vector.chunk_tokens[chunk]) {
                assembler.apply(tracker.feed(token), index);
            }

            // Notes from chunk k-1 are only final once chunk k has closed.
            if (index > 0) {
                const std::vector<Note> ready = assembler.closedIn(index - 1);
                streamed.insert(streamed.end(), ready.begin(), ready.end());
            }
        }

        assembler.apply(tracker.finish(), last_chunk);

        // The withheld tail is the last chunk only -- every earlier one was
        // released as soon as its successor closed. Emitting the second-to-last
        // again here would duplicate it.
        //
        // Notes that finish() closes cannot trim an earlier chunk's: they were
        // open the whole time, so nothing else can have opened on their channel.
        const std::vector<Note> tail = assembler.closedIn(last_chunk);
        streamed.insert(streamed.end(), tail.begin(), tail.end());

        std::vector<Note> want = assembler.finalize();
        sortNotes(streamed);

        INFO("streamed:\n" << describeNotes(streamed) << "final:\n" << describeNotes(want));
        REQUIRE(streamed.size() == want.size());

        for (std::size_t i = 0; i < want.size(); ++i) {
            INFO("note " << i);
            CHECK(streamed[i].pitch == want[i].pitch);
            CHECK(streamed[i].program == want[i].program);
            CHECK(streamed[i].is_drum == want[i].is_drum);
            CHECK_THAT(streamed[i].onset, Catch::Matchers::WithinAbs(want[i].onset, 1e-12));
            CHECK_THAT(streamed[i].offset, Catch::Matchers::WithinAbs(want[i].offset, 1e-12));
        }
    }
}

TEST_CASE("the horizon only claims what has already been streamed", "[pure][notes]")
{
    // TranscriptionUpdate::finalized_through is what lets a host play a
    // transcription while it is still being decoded, so it has to be an
    // under-claim in the worst case and never an over-claim: at the call after
    // chunk k, every note ending below that chunk's seek time is already out.
    for (const NoteVector& vector: Vectors::get().noteVectors()) {
        INFO(vector.name);

        OpenNoteTracker tracker;
        NoteAssembler assembler;
        const std::vector<ChunkBoundary> boundaries = vector.boundaries();
        const int last_chunk = static_cast<int>(boundaries.size()) - 1;

        std::vector<Note> streamed;

        // Each call's horizon, paired with how much had been streamed by then.
        std::vector<std::pair<double, std::size_t>> calls;

        for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
            const int index = static_cast<int>(chunk);
            assembler.apply(tracker.feed(boundaries[chunk]), index);

            for (const std::int32_t token: vector.chunk_tokens[chunk]) {
                assembler.apply(tracker.feed(token), index);
            }

            if (index > 0) {
                const std::vector<Note> ready = assembler.closedIn(index - 1);
                streamed.insert(streamed.end(), ready.begin(), ready.end());
            }

            // What transcribe() reports here: chunks 0..k-1 are out, and each
            // closed only notes ending inside its own window.
            calls.emplace_back(boundaries[chunk].seek_time, streamed.size());
        }

        // Read before finish(), which clears them: notes the model never closed.
        // finish() gives each a 10 ms offset that can land far behind the last
        // horizon, and they surface only in the final call. That is the one
        // documented exception to the promise being checked here.
        const std::vector<NoteKey> never_closed = tracker.openKeys();

        assembler.apply(tracker.finish(), last_chunk);
        const std::vector<Note> tail = assembler.closedIn(last_chunk);
        streamed.insert(streamed.end(), tail.begin(), tail.end());

        // Open notes are melodic -- a drum hit closes the instant it opens -- so
        // the note carries the key's program rather than DRUM_PROGRAM.
        const auto neverClosed = [&](const Note& inNote) {
            return std::any_of(never_closed.begin(), never_closed.end(), [&](const NoteKey& inKey) {
                return inKey.program == inNote.program && inKey.pitch == inNote.pitch;
            });
        };

        for (const auto& [horizon, streamed_by_then]: calls) {
            INFO("horizon " << horizon);

            for (std::size_t i = streamed_by_then; i < streamed.size(); ++i) {
                INFO("arrived later: onset " << streamed[i].onset << ", offset " << streamed[i].offset << ", pitch "
                                             << streamed[i].pitch);
                CHECK((streamed[i].offset >= horizon || neverClosed(streamed[i])));
            }
        }
    }
}
