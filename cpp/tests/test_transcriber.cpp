// The public API, end to end: waveform in, notes out.
//
// Everything under it is already validated stage by stage, so what is left for
// this file is the facade's own contract -- chunking and absolute time, the
// streaming callback's promises, cancellation, error mapping, and that each
// configuration reproduces the reference's notes.
//
// Each configuration transcribes the fixture once (inference.hpp), with its
// callback updates recorded, and the cases below read that one result. The
// signal is reassembled from the reference's own dumped chunk waveforms, so the
// suite still needs no WAV parser.

#include "inference.hpp"

#include "note_assembler.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace msl;
using namespace msl::test;

namespace
{

constexpr double TIME_TOLERANCE = 1e-9;

bool sameNote(const Note& inA, const Note& inB)
{
    return inA.pitch == inB.pitch && inA.program == inB.program && inA.is_drum == inB.is_drum
           && std::abs(inA.onset - inB.onset) <= TIME_TOLERANCE && std::abs(inA.offset - inB.offset) <= TIME_TOLERANCE;
}

void requireSameNotes(const std::vector<Note>& inGot, const std::vector<Note>& inWant)
{
    INFO("got:\n" << describeNotes(inGot) << "want:\n" << describeNotes(inWant));
    REQUIRE(inGot.size() == inWant.size());

    for (std::size_t i = 0; i < inWant.size(); ++i) {
        INFO("note " << i);
        REQUIRE(sameNote(inGot[i], inWant[i]));
    }
}

/**
 * A free-running transcription against a variant's reference notes.
 *
 * Exact when the variant's teacher-forced decode agrees with the reference at
 * every step, because the free-running decode then feeds the same tokens. A
 * near-tie in chunk k lets the two streams part from there, so the bar becomes
 * everything that was settled before it: every note ending before chunk k's
 * start was decided by tokens both runs share, and has to match exactly.
 */
void requireReferenceNotes(const std::string& inVariant)
{
    const VariantReference& variant = requireReferences().variant(inVariant);
    const Transcription& run = transcription(inVariant);
    REQUIRE(run.notes.has_value());

    int first_near_tie_chunk = -1;
    int near_ties = 0;

    for (int chunk = 0; chunk < static_cast<int>(variant.tokens.size()); ++chunk) {
        INFO("chunk " << chunk);
        const int in_chunk = requireOnlyNearTies(variantChunk(inVariant, chunk));
        near_ties += in_chunk;

        if (in_chunk > 0 && first_near_tie_chunk < 0) {
            first_near_tie_chunk = chunk;
        }
    }

    if (first_near_tie_chunk < 0) {
        requireSameNotes(*run.notes, variant.notes);
        return;
    }

    const double settled = first_near_tie_chunk * Transcriber::SEGMENT_DURATION;
    const auto settledBefore = [settled](const std::vector<Note>& inNotes) {
        std::vector<Note> out;
        std::copy_if(inNotes.begin(), inNotes.end(), std::back_inserter(out), [settled](const Note& inNote) {
            return inNote.offset < settled;
        });
        return out;
    };

    INFO("first near-tie in chunk " << first_near_tie_chunk << "; comparing notes that end before " << settled << " s");
    requireSameNotes(settledBefore(*run.notes), settledBefore(variant.notes));

    WARN(msl::format("{} argmax near-tie(s), the first in chunk {}: notes after {} s may differ from the "
                     "reference ({} against {} in total)",
                     near_ties,
                     first_near_tie_chunk,
                     formatDouble("%.1f", settled),
                     run.notes->size(),
                     variant.notes.size()));
}

} // namespace

TEST_CASE("chunking matches the reference's ceil division", "[transcriber]")
{
    CHECK(Transcriber::chunkCount(0) == 0);
    CHECK(Transcriber::chunkCount(1) == 1);
    CHECK(Transcriber::chunkCount(Transcriber::SEGMENT_SAMPLES) == 1);
    CHECK(Transcriber::chunkCount(Transcriber::SEGMENT_SAMPLES + 1) == 2);
    CHECK(Transcriber::chunkCount(3 * Transcriber::SEGMENT_SAMPLES) == 3);
}

TEST_CASE("loading reports why it failed rather than throwing", "[transcriber]")
{
    // A host picks the checkpoint path from a file dialog or a download cache,
    // so a bad one is an outcome to show the user, not an exception.
    const std::expected<Transcriber, Error> missing = Transcriber::load("does-not-exist.gguf");
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error() == Error::FileNotFound);

    const std::expected<Transcriber, Error> not_a_gguf =
        Transcriber::load(testdataRoot() / "audio" / "fixture_3chunks_16k.wav");
    REQUIRE_FALSE(not_a_gguf.has_value());
    CHECK(not_a_gguf.error() == Error::InvalidCheckpoint);
}

TEST_CASE("an empty signal transcribes to no notes", "[transcriber]")
{
    const std::expected<std::vector<Note>, Error> notes = sharedTranscriber().transcribe({});
    REQUIRE(notes.has_value());
    CHECK(notes->empty());
}

TEST_CASE("an instrument that has no name is rejected as the caller's mistake", "[transcriber]")
{
    // Distinguishable from Error::Internal on purpose: a host handed a bad
    // selection can fix its own call, and telling it "internal error" sends it
    // to file a bug against this library instead.
    TranscribeOptions options;
    options.instruments = {static_cast<InstrumentGroup>(999)};

    const std::expected<std::vector<Note>, Error> bogus = sharedTranscriber().transcribe({}, options);
    REQUIRE_FALSE(bogus.has_value());
    CHECK(bogus.error() == Error::InvalidArgument);

    // Ids 34 and 35 are real groups upstream that were simply never named, so
    // they are the selection a caller is most likely to arrive at honestly --
    // by counting group ids rather than going through allInstrumentGroups().
    for (const std::int32_t id: {34, 35}) {
        INFO("group id " << id);
        options.instruments = {static_cast<InstrumentGroup>(id)};

        const std::expected<std::vector<Note>, Error> unnamed = sharedTranscriber().transcribe({}, options);
        REQUIRE_FALSE(unnamed.has_value());
        CHECK(unnamed.error() == Error::InvalidArgument);
    }

    // Every group the library does offer is accepted, so the check rejects the
    // unnamed ones and nothing else.
    options.instruments.assign(allInstrumentGroups().begin(), allInstrumentGroups().end());
    CHECK(sharedTranscriber().transcribe({}, options).has_value());
}

TEST_CASE("returning false from the callback cancels", "[transcriber]")
{
    // Two chunks of silence: the callback fires after the first, and silence
    // decodes in a handful of tokens.
    const std::vector<float> silence(2 * Transcriber::SEGMENT_SAMPLES, 0.0f);
    int calls = 0;

    const std::expected<std::vector<Note>, Error> result =
        sharedTranscriber().transcribe(silence, {}, [&](const TranscriptionUpdate&) {
            ++calls;
            return false;
        });

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == Error::Cancelled);
    CHECK(calls == 1);

    // And the instance is reusable: state is cleared on entry, not on exit, so a
    // cancelled run leaves nothing for the next one to inherit.
    const std::expected<std::vector<Note>, Error> again = sharedTranscriber().transcribe({});
    REQUIRE(again.has_value());
    CHECK(again->empty());
}

TEST_CASE("the default transcription reproduces the reference notes", "[transcriber][slow]")
{
    // Defaults mean prelude forcing on and no instrument filter, which is the
    // `prelude` variant -- so this is the end-to-end statement the whole library
    // exists to make: a waveform in, and PyTorch's own note list back.
    const VariantReference& variant = requireReferences().variant("prelude");
    REQUIRE(variant.prelude_forcing);
    REQUIRE(variant.instruments.empty());

    const TranscribeOptions defaults;
    CHECK(transcription("prelude").options.prelude_forcing == defaults.prelude_forcing);

    requireReferenceNotes("prelude");
}

TEST_CASE("an unforced transcription reproduces the reference notes", "[transcriber][slow]")
{
    // The waveform path through every chunk: the port's own STFT, then each
    // chunk decoded independently.
    requireReferenceNotes("plain");
}

TEST_CASE("times are absolute in the signal, not per chunk", "[transcriber][slow]")
{
    // The easiest way for chunking to be wrong and still look plausible: every
    // chunk decodes fine but nothing is offset, so the whole transcription piles
    // up in the first five seconds.
    const Transcription& run = transcription("prelude");
    REQUIRE(run.notes.has_value());
    REQUIRE_FALSE(run.notes->empty());

    const double duration = static_cast<double>(fixtureSignal().size()) / Transcriber::SAMPLE_RATE;
    const double last_onset = run.notes->back().onset;

    CHECK(last_onset > 2.0 * Transcriber::SEGMENT_DURATION);
    CHECK(last_onset < duration);

    // And sorted, which is the documented order.
    CHECK(std::is_sorted(
        run.notes->begin(), run.notes->end(), [](const Note& a, const Note& b) { return a.onset < b.onset; }));
}

TEST_CASE("the callback streams every note exactly once, in order", "[transcriber][slow]")
{
    // The contract a progressively-filling piano roll depends on: append-only,
    // no duplicates, no later corrections.
    const Transcription& run = transcription("prelude");
    REQUIRE(run.notes.has_value());
    REQUIRE_FALSE(run.updates.empty());

    std::vector<Note> streamed;
    std::vector<float> progress;
    std::vector<double> horizons;

    // How much had been streamed by the time each horizon was reported, so the
    // "everything below it is already out" claim can be checked after the fact.
    std::vector<std::size_t> streamed_at_horizon;

    for (const RecordedUpdate& update: run.updates) {
        streamed.insert(streamed.end(), update.new_notes.begin(), update.new_notes.end());
        progress.push_back(update.progress);
        horizons.push_back(update.finalized_through);
        streamed_at_horizon.push_back(streamed.size());
    }

    CHECK(std::is_sorted(progress.begin(), progress.end()));
    CHECK_THAT(progress.back(), Catch::Matchers::WithinAbs(1.0, 1e-6));

    CHECK(std::is_sorted(horizons.begin(), horizons.end()));

    // The horizon's promise: at the moment it was reported, every note ending
    // below it had already been streamed. Notes the model never closed are the
    // documented exception -- finish() gives them a 10 ms offset that can land
    // far behind the line -- so they are excluded by matching against what was
    // streamed rather than against the final list.
    for (std::size_t call = 0; call < horizons.size(); ++call) {
        INFO("call " << call << ", horizon " << horizons[call]);

        const std::size_t below = static_cast<std::size_t>(std::count_if(
            streamed.begin(), streamed.end(), [&](const Note& inNote) { return inNote.offset < horizons[call]; }));

        // Everything that ends below the horizon was among the first
        // streamed_at_horizon[call] notes -- nothing below it arrived later.
        const std::size_t below_at_the_time = static_cast<std::size_t>(
            std::count_if(streamed.begin(),
                          streamed.begin() + static_cast<std::ptrdiff_t>(streamed_at_horizon[call]),
                          [&](const Note& inNote) { return inNote.offset < horizons[call]; }));

        CHECK(below == below_at_the_time);
    }

    sortNotes(streamed);
    requireSameNotes(streamed, *run.notes);
}

TEST_CASE("an instrument selection restricts what comes out", "[transcriber][slow]")
{
    const Transcription& run = transcription("band");
    const std::vector<InstrumentGroup>& selected = run.options.instruments;

    REQUIRE(run.notes.has_value());
    REQUIRE_FALSE(run.notes->empty());

    for (const Note& note: *run.notes) {
        INFO("program " << note.program << " (" << instrumentLabel(note.program) << ")");
        const std::optional<InstrumentGroup> group = instrumentGroupFor(note.program);
        REQUIRE(group.has_value());
        REQUIRE(std::find(selected.begin(), selected.end(), *group) != selected.end());
    }

    // Drums and bass both come through, so the filter is restricting rather than
    // collapsing everything onto one instrument.
    const bool has_drums = std::any_of(run.notes->begin(), run.notes->end(), [](const Note& n) { return n.is_drum; });
    const bool has_bass = std::any_of(run.notes->begin(), run.notes->end(), [](const Note& n) {
        return n.program == programFor(InstrumentGroup::ElectricBass);
    });

    CHECK(has_drums);
    CHECK(has_bass);
}

TEST_CASE("turning prelude forcing off changes the result", "[transcriber][slow]")
{
    // Not a quality claim, just that the option is wired: with forcing off the
    // model writes its own tie prologue at every seam instead of being handed
    // one, and the transcription differs.
    const Transcription& forced = transcription("prelude");
    const Transcription& unforced = transcription("plain");

    REQUIRE(forced.notes.has_value());
    REQUIRE(unforced.notes.has_value());

    const bool identical = forced.notes->size() == unforced.notes->size()
                           && std::equal(forced.notes->begin(), forced.notes->end(), unforced.notes->begin(), sameNote);
    CHECK_FALSE(identical);
}
