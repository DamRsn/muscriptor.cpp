// The decode state machine and note assembly, against hand-authored vectors.
//
// Each vector is a token sequence decoded by the reference's own
// OpenNoteTracker, recorded as both actions and notes, so a disagreement names
// which of the two ports broke. They cover rules the fixture's real output
// never reaches: a shift before a tie, shift 0, a non-monotonic shift, and an
// event past the window.
//
// Runs with no weights and no reference dump -- tagged [pure].

#include "vectors.hpp"

#include "note_assembler.hpp"
#include "open_note_tracker.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>

using namespace msl;
using namespace msl::test;

namespace
{

/** Times are derived identically on both sides, so this is tight on purpose. */
constexpr double TIME_TOLERANCE = 1e-12;

void checkNotesEqual(const std::vector<Note>& inGot, const std::vector<Note>& inWant)
{
    INFO("got:\n" << describeNotes(inGot) << "want:\n" << describeNotes(inWant));
    REQUIRE(inGot.size() == inWant.size());

    for (std::size_t i = 0; i < inWant.size(); ++i) {
        INFO("note " << i);
        CHECK(inGot[i].pitch == inWant[i].pitch);
        CHECK(inGot[i].program == inWant[i].program);
        CHECK(inGot[i].is_drum == inWant[i].is_drum);
        CHECK_THAT(inGot[i].onset, Catch::Matchers::WithinAbs(inWant[i].onset, TIME_TOLERANCE));
        CHECK_THAT(inGot[i].offset, Catch::Matchers::WithinAbs(inWant[i].offset, TIME_TOLERANCE));
    }
}

/** Replays one vector, returning its actions in order. */
std::vector<NoteAction> replayActions(const NoteVector& inVector)
{
    OpenNoteTracker tracker;
    std::vector<NoteAction> actions;
    const std::vector<ChunkBoundary> boundaries = inVector.boundaries();

    for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
        const std::vector<NoteAction> at_boundary = tracker.feed(boundaries[chunk]);
        actions.insert(actions.end(), at_boundary.begin(), at_boundary.end());

        for (const std::int32_t token: inVector.chunk_tokens[chunk]) {
            const std::vector<NoteAction> from_token = tracker.feed(token);
            actions.insert(actions.end(), from_token.begin(), from_token.end());
        }
    }

    const std::vector<NoteAction> at_end = tracker.finish();
    actions.insert(actions.end(), at_end.begin(), at_end.end());
    return actions;
}

/** Replays one vector all the way to cleaned notes. */
std::vector<Note> replayNotes(const NoteVector& inVector)
{
    OpenNoteTracker tracker;
    NoteAssembler assembler;
    const std::vector<ChunkBoundary> boundaries = inVector.boundaries();

    for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
        const int index = static_cast<int>(chunk);
        assembler.apply(tracker.feed(boundaries[chunk]), index);

        for (const std::int32_t token: inVector.chunk_tokens[chunk]) {
            assembler.apply(tracker.feed(token), index);
        }
    }

    assembler.apply(tracker.finish(), static_cast<int>(boundaries.size()) - 1);
    return assembler.finalize();
}

} // namespace

TEST_CASE("every vector produces the reference's action stream", "[pure][tracker]")
{
    // The bisection point. If this passes and the note test below fails, the
    // state machine is fine and the assembler is not.
    for (const NoteVector& vector: Vectors::get().noteVectors()) {
        INFO(vector.name << "\n" << vector.description);

        const std::vector<NoteAction> got = replayActions(vector);
        INFO("got:\n" << describeActions(got) << "want:\n" << describeActions(vector.actions));
        REQUIRE(got.size() == vector.actions.size());

        for (std::size_t i = 0; i < vector.actions.size(); ++i) {
            INFO("action " << i);
            CHECK(got[i].kind == vector.actions[i].kind);
            CHECK(got[i].pitch == vector.actions[i].pitch);
            CHECK_THAT(got[i].time, Catch::Matchers::WithinAbs(vector.actions[i].time, TIME_TOLERANCE));

            // Drum hits carry no program in the reference's action type.
            if (got[i].kind != NoteActionKind::DrumHit) {
                CHECK(got[i].program == vector.actions[i].program);
            }
        }
    }
}

TEST_CASE("every vector produces the reference's note list", "[pure][tracker]")
{
    for (const NoteVector& vector: Vectors::get().noteVectors()) {
        INFO(vector.name << "\n" << vector.description);
        checkNotesEqual(replayNotes(vector), vector.notes);
    }
}

TEST_CASE("open keys at each boundary match the reference", "[pure][tracker]")
{
    // This is what prelude forcing reads. Checking it separately means a wrong
    // forced prologue localises here, with no transformer involved.
    for (const NoteVector& vector: Vectors::get().noteVectors()) {
        INFO(vector.name);

        OpenNoteTracker tracker;
        const std::vector<ChunkBoundary> boundaries = vector.boundaries();
        REQUIRE(vector.open_keys_at_boundary.size() == boundaries.size());

        for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
            INFO("boundary " << chunk);
            tracker.feed(boundaries[chunk]);

            // Read after feeding the boundary, never before: the boundary
            // settles a previous chunk that ended mid-prologue, and only then
            // is this the decoder's own view.
            CHECK(tracker.openKeys() == vector.open_keys_at_boundary[chunk]);

            for (const std::int32_t token: vector.chunk_tokens[chunk]) {
                tracker.feed(token);
            }
        }
    }
}

TEST_CASE("the open set keeps insertion order, not key order", "[pure][tracker]")
{
    // finish() replays its closes in insertion order and that order reaches the
    // caller. A std::map keyed on (program, pitch) would silently reorder them
    // and every other test here would still pass.
    const NoteVector& vector = Vectors::get().noteVector("insertion_order_survives_to_finish");
    const std::vector<NoteAction> actions = replayActions(vector);

    std::vector<NoteAction> ends;
    std::copy_if(actions.begin(), actions.end(), std::back_inserter(ends), [](const NoteAction& a) {
        return a.kind == NoteActionKind::End;
    });

    REQUIRE(ends.size() == 3);
    CHECK(ends[0].program == 33);
    CHECK(ends[1].program == 0);
    CHECK(ends[2].program == 0);
    CHECK(ends[1].pitch == 60);
    CHECK(ends[2].pitch == 64);
}

TEST_CASE("a reset tracker decodes as a fresh one", "[pure][tracker]")
{
    const NoteVector& vector = Vectors::get().noteVector("tie_prologue_partial");

    OpenNoteTracker tracker;
    tracker.feed(ChunkBoundary {0.0, 5.0});
    for (const std::int32_t token: vector.chunk_tokens[0]) {
        tracker.feed(token);
    }

    REQUIRE_FALSE(tracker.openKeys().empty());
    tracker.reset();
    CHECK(tracker.openKeys().empty());

    // And nothing carries over: finish() on a reset tracker has nothing to say.
    CHECK(tracker.finish().empty());
}
