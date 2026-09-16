// Real model output, all the way to notes, with no inference.
//
// The reference's token streams over the audio fixture, for every note-level
// variant, replayed through the C++ tracker and assembler. The hand-authored
// vectors in test_tracker.cpp cover the rules real output never reaches.
//
// Needs testdata/refs/<size>/notes.json but no weights.

#include "vectors.hpp"

#include "note_assembler.hpp"
#include "open_note_tracker.hpp"
#include "vocabulary.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <set>

using namespace msl;
using namespace msl::test;

namespace
{

constexpr double TIME_TOLERANCE = 1e-12;

struct Replay {
    std::vector<NoteAction> actions;
    std::vector<Note> notes;
    std::vector<std::vector<NoteKey>> open_keys_at_boundary;
};

Replay replay(const VariantReference& inVariant)
{
    OpenNoteTracker tracker;
    NoteAssembler assembler;
    Replay out;

    const std::vector<ChunkBoundary> boundaries = inVariant.boundaries();

    for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
        const int index = static_cast<int>(chunk);

        const std::vector<NoteAction> at_boundary = tracker.feed(boundaries[chunk]);
        out.actions.insert(out.actions.end(), at_boundary.begin(), at_boundary.end());
        assembler.apply(at_boundary, index);

        // Read after the boundary is fed: this is what prelude forcing sees.
        out.open_keys_at_boundary.push_back(tracker.openKeys());

        for (const std::int32_t token: inVariant.tokens[chunk]) {
            const std::vector<NoteAction> from_token = tracker.feed(token);
            out.actions.insert(out.actions.end(), from_token.begin(), from_token.end());
            assembler.apply(from_token, index);
        }
    }

    const std::vector<NoteAction> at_end = tracker.finish();
    out.actions.insert(out.actions.end(), at_end.begin(), at_end.end());
    assembler.apply(at_end, static_cast<int>(boundaries.size()) - 1);

    out.notes = assembler.finalize();
    return out;
}

} // namespace

TEST_CASE("real token streams produce the reference's actions", "[tokens-to-notes]")
{
    for (const VariantReference& variant: requireReferences().variants()) {
        INFO("variant " << variant.name);

        const Replay got = replay(variant);
        REQUIRE(got.actions.size() == variant.actions.size());

        for (std::size_t i = 0; i < variant.actions.size(); ++i) {
            INFO("action " << i);
            CHECK(got.actions[i].kind == variant.actions[i].kind);
            CHECK(got.actions[i].pitch == variant.actions[i].pitch);
            CHECK_THAT(got.actions[i].time, Catch::Matchers::WithinAbs(variant.actions[i].time, TIME_TOLERANCE));

            if (got.actions[i].kind != NoteActionKind::DrumHit) {
                CHECK(got.actions[i].program == variant.actions[i].program);
            }
        }
    }
}

TEST_CASE("real token streams produce the reference's notes", "[tokens-to-notes]")
{
    for (const VariantReference& variant: requireReferences().variants()) {
        INFO("variant " << variant.name);

        const Replay got = replay(variant);
        REQUIRE(got.notes.size() == variant.notes.size());

        for (std::size_t i = 0; i < variant.notes.size(); ++i) {
            INFO("note " << i);
            CHECK(got.notes[i].pitch == variant.notes[i].pitch);
            CHECK(got.notes[i].program == variant.notes[i].program);
            CHECK(got.notes[i].is_drum == variant.notes[i].is_drum);
            CHECK_THAT(got.notes[i].onset, Catch::Matchers::WithinAbs(variant.notes[i].onset, TIME_TOLERANCE));
            CHECK_THAT(got.notes[i].offset, Catch::Matchers::WithinAbs(variant.notes[i].offset, TIME_TOLERANCE));
        }
    }
}

TEST_CASE("the forced prologue is what our own open keys encode to", "[tokens-to-notes][prelude]")
{
    // Prelude forcing without the transformer: at each boundary the reference
    // read its tracker's open keys and encoded them, and the tokens it produced
    // are recorded. If openKeys() or tieSectionTokenIds() is wrong, this fails
    // here rather than as a diverging token stream three stages later.
    const VariantReference& variant = requireReferences().variant("prelude");
    REQUIRE(variant.prelude_forcing);

    const Replay got = replay(variant);
    REQUIRE(got.open_keys_at_boundary.size() == variant.open_keys_at_boundary.size());

    for (std::size_t chunk = 0; chunk < variant.open_keys_at_boundary.size(); ++chunk) {
        INFO("boundary " << chunk);
        REQUIRE(got.open_keys_at_boundary[chunk] == variant.open_keys_at_boundary[chunk]);

        if (chunk == 0) {
            // The first chunk has nothing to carry, so it is never forced.
            CHECK(variant.prompts[chunk].empty());
            continue;
        }

        const std::vector<std::int32_t> prompt = Vocabulary::tieSectionTokenIds(got.open_keys_at_boundary[chunk]);
        CHECK(prompt == variant.prompts[chunk]);

        // And the chunk's own stream really does begin with it -- which is what
        // makes feeding the prompt through the tracker a no-op by construction.
        REQUIRE(variant.tokens[chunk].size() >= prompt.size());
        CHECK(std::vector<std::int32_t>(variant.tokens[chunk].begin(),
                                        variant.tokens[chunk].begin() + static_cast<std::ptrdiff_t>(prompt.size()))
              == prompt);
    }
}

TEST_CASE("prelude forcing changes the transcription", "[tokens-to-notes][prelude]")
{
    // Cheap, but worth pinning: if forcing ever silently stopped applying, the
    // suite above would keep passing on identical streams and the feature would
    // be gone.
    const NoteReferences& refs = requireReferences();
    CHECK(refs.variant("prelude").tokens != refs.variant("plain").tokens);
}

TEST_CASE("an instrument filter keeps its promise", "[tokens-to-notes][instruments]")
{
    // The hard half of instrument selection is a guarantee, not a hint: no
    // forbidden token can appear, so no other instrument can.
    const NoteReferences& refs = requireReferences();

    for (const std::string& name: {"bass", "band"}) {
        const VariantReference& variant = refs.variant(name);
        INFO("variant " << name);
        REQUIRE_FALSE(variant.forbidden_token_ids.empty());

        const std::set<std::int32_t> forbidden(variant.forbidden_token_ids.begin(), variant.forbidden_token_ids.end());

        for (const std::vector<std::int32_t>& chunk: variant.tokens) {
            for (const std::int32_t token: chunk) {
                REQUIRE_FALSE(forbidden.contains(token));
            }
        }

        // And nothing outside the selection survives into the notes.
        std::set<int> programs;
        for (const Note& note: replay(variant).notes) {
            programs.insert(note.program);
        }

        for (const int program: programs) {
            INFO("decoded program " << program);
            const std::optional<InstrumentGroup> group = instrumentGroupFor(program);
            REQUIRE(group.has_value());
            CHECK(std::find(variant.instruments.begin(), variant.instruments.end(), std::string(instrumentName(*group)))
                  != variant.instruments.end());
        }
    }
}

TEST_CASE("the streaming slice reproduces the full note list on real output", "[tokens-to-notes]")
{
    // Same contract as the vector-level test, over real output: what the
    // per-chunk callback reports, concatenated, is what transcribe() returns.
    for (const VariantReference& variant: requireReferences().variants()) {
        INFO("variant " << variant.name);

        OpenNoteTracker tracker;
        NoteAssembler assembler;
        const std::vector<ChunkBoundary> boundaries = variant.boundaries();
        const int last_chunk = static_cast<int>(boundaries.size()) - 1;

        std::vector<Note> streamed;

        // Each call's finalized_through, paired with how much had been streamed
        // by then, so the horizon's claim can be checked against real output.
        std::vector<std::pair<double, std::size_t>> calls;

        for (std::size_t chunk = 0; chunk < boundaries.size(); ++chunk) {
            const int index = static_cast<int>(chunk);
            assembler.apply(tracker.feed(boundaries[chunk]), index);

            for (const std::int32_t token: variant.tokens[chunk]) {
                assembler.apply(tracker.feed(token), index);
            }

            if (index > 0) {
                const std::vector<Note> ready = assembler.closedIn(index - 1);
                streamed.insert(streamed.end(), ready.begin(), ready.end());
            }

            calls.emplace_back(boundaries[chunk].seek_time, streamed.size());
        }

        // Read before finish() clears them; see the vector-level horizon test.
        const std::vector<NoteKey> never_closed = tracker.openKeys();

        assembler.apply(tracker.finish(), last_chunk);
        const std::vector<Note> tail = assembler.closedIn(last_chunk);
        streamed.insert(streamed.end(), tail.begin(), tail.end());

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

        sortNotes(streamed);
        const std::vector<Note> want = assembler.finalize();

        REQUIRE(streamed.size() == want.size());

        for (std::size_t i = 0; i < want.size(); ++i) {
            INFO("note " << i);
            CHECK(streamed[i].pitch == want[i].pitch);
            CHECK(streamed[i].program == want[i].program);
            CHECK(streamed[i].is_drum == want[i].is_drum);
            CHECK_THAT(streamed[i].onset, Catch::Matchers::WithinAbs(want[i].onset, TIME_TOLERANCE));
            CHECK_THAT(streamed[i].offset, Catch::Matchers::WithinAbs(want[i].offset, TIME_TOLERANCE));
        }
    }
}
