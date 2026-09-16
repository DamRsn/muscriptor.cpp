// The MT3 vocabulary: a fixed arithmetic id<->event mapping, checked against
// the reference's own table.
//
// There is nothing numerical here, which is exactly why it is worth being
// thorough: an off-by-one in a range bound shifts an entire event class and
// produces a token stream that decodes into plausible, wrong notes. Every one
// of the 1393 ids is round-tripped rather than sampled.
//
// Runs with no weights and no reference dump -- tagged [pure].

#include "vectors.hpp"

#include "vocabulary.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <map>

using namespace msl;
using namespace msl::test;

namespace
{

EventType typeForName(const std::string& inName)
{
    static const std::map<std::string, EventType> BY_NAME {
        {"PAD", EventType::Pad},
        {"EOS", EventType::Eos},
        {"UNK", EventType::Unk},
        {"shift", EventType::Shift},
        {"pitch", EventType::Pitch},
        {"velocity", EventType::Velocity},
        {"tie", EventType::Tie},
        {"program", EventType::Program},
        {"drum", EventType::Drum},
    };

    const auto it = BY_NAME.find(inName);
    REQUIRE(it != BY_NAME.end());
    return it->second;
}

} // namespace

TEST_CASE("the vocabulary has the reference's size and special ids", "[pure][vocabulary]")
{
    const Vectors& vectors = Vectors::get();

    CHECK(Vocabulary::NUM_TOKENS == vectors.numTokens());
    CHECK(Vocabulary::EOS_ID == vectors.eosId());
}

TEST_CASE("every event range starts and ends where the reference's does", "[pure][vocabulary]")
{
    const Vectors& vectors = Vectors::get();

    struct Expected {
        const char* name;
        EventType type;
        std::int32_t first;
        std::int32_t count;
    };

    const std::array<Expected, 9> ranges {{
        {"PAD", EventType::Pad, Vocabulary::PAD_ID, 1},
        {"EOS", EventType::Eos, Vocabulary::EOS_ID, 1},
        {"UNK", EventType::Unk, Vocabulary::UNK_ID, 1},
        {"shift", EventType::Shift, Vocabulary::SHIFT_FIRST, Vocabulary::SHIFT_COUNT},
        {"pitch", EventType::Pitch, Vocabulary::PITCH_FIRST, Vocabulary::PITCH_COUNT},
        {"velocity", EventType::Velocity, Vocabulary::VELOCITY_FIRST, Vocabulary::VELOCITY_COUNT},
        {"tie", EventType::Tie, Vocabulary::TIE_FIRST, Vocabulary::TIE_COUNT},
        {"program", EventType::Program, Vocabulary::PROGRAM_FIRST, Vocabulary::PROGRAM_COUNT},
        {"drum", EventType::Drum, Vocabulary::DRUM_FIRST, Vocabulary::DRUM_COUNT},
    }};

    for (const Expected& expected: ranges) {
        INFO("event type " << expected.name);
        const auto [first, last] = vectors.vocabRange(expected.name);
        CHECK(expected.first == first);
        CHECK(expected.first + expected.count - 1 == last);
    }
}

TEST_CASE("every token id round-trips through eventFor and tokenFor", "[pure][vocabulary]")
{
    for (std::int32_t id = 0; id < Vocabulary::NUM_TOKENS; ++id) {
        INFO("token " << id);
        const TokenEvent event = Vocabulary::eventFor(id);
        REQUIRE(Vocabulary::tokenFor(event.type, event.value) == id);
    }
}

TEST_CASE("range boundaries decode to what the reference says", "[pure][vocabulary]")
{
    // The spot checks straddle every range boundary, so a range that moved by
    // one names itself here instead of showing up as a wrong note much later.
    for (const VocabSpotCheck& check: Vectors::get().vocabSpotChecks()) {
        INFO("token " << check.token_id << " should be " << check.type << " " << check.value);
        const TokenEvent event = Vocabulary::eventFor(check.token_id);
        CHECK(event.type == typeForName(check.type));
        CHECK(event.value == check.value);
    }
}

TEST_CASE("ids outside the vocabulary decode to Unk rather than aliasing", "[pure][vocabulary]")
{
    // The medium and large checkpoints carry card = 1395, two ids past the
    // vocabulary. _compute_logits masks them so they can never be sampled, but
    // the decoder must not fold them onto a real event if one ever arrives.
    for (const std::int32_t id: {-1, Vocabulary::NUM_TOKENS, Vocabulary::NUM_TOKENS + 1, 1'000'000}) {
        INFO("token " << id);
        CHECK(Vocabulary::eventFor(id).type == EventType::Unk);
    }
}

TEST_CASE("out-of-range values have no token", "[pure][vocabulary]")
{
    CHECK(Vocabulary::tokenFor(EventType::Pitch, -1) == -1);
    CHECK(Vocabulary::tokenFor(EventType::Pitch, Vocabulary::PITCH_COUNT) == -1);
    CHECK(Vocabulary::tokenFor(EventType::Shift, Vocabulary::MAX_SHIFT_STEPS) == -1);
    CHECK(Vocabulary::tokenFor(EventType::Velocity, 2) == -1);
    CHECK(Vocabulary::tokenFor(EventType::Program, Vocabulary::PROGRAM_COUNT) == -1);
}

TEST_CASE("tie prologues encode exactly as the reference encodes them", "[pure][vocabulary]")
{
    for (const TieSectionCase& expected: Vectors::get().tieSectionCases()) {
        INFO("open keys: " << expected.open_keys.size());
        CHECK(Vocabulary::tieSectionTokenIds(expected.open_keys) == expected.token_ids);
    }
}

TEST_CASE("an empty tie prologue is still a tie token", "[pure][vocabulary]")
{
    // Chunks with nothing sustained still have to declare that, or the decoder
    // reads the chunk as malformed and throws it away.
    const std::vector<std::int32_t> tokens = Vocabulary::tieSectionTokenIds({});
    REQUIRE(tokens.size() == 1);
    CHECK(Vocabulary::eventFor(tokens[0]).type == EventType::Tie);
}

TEST_CASE("the tie prologue sorts its input and makes program sticky", "[pure][vocabulary]")
{
    // Same keys, scrambled: the encoding must not depend on the order the
    // tracker happened to hand them over in, and one program token has to cover
    // its whole run of pitches.
    const std::array<NoteKey, 4> scrambled {{{33, 40}, {0, 67}, {0, 60}, {33, 38}}};
    const std::array<NoteKey, 4> sorted {{{0, 60}, {0, 67}, {33, 38}, {33, 40}}};

    const std::vector<std::int32_t> tokens = Vocabulary::tieSectionTokenIds(scrambled);
    CHECK(tokens == Vocabulary::tieSectionTokenIds(sorted));

    // program, pitch, pitch, program, pitch, pitch, tie
    REQUIRE(tokens.size() == 7);
    CHECK(Vocabulary::eventFor(tokens[0]) == TokenEvent {EventType::Program, 0});
    CHECK(Vocabulary::eventFor(tokens[1]) == TokenEvent {EventType::Pitch, 60});
    CHECK(Vocabulary::eventFor(tokens[2]) == TokenEvent {EventType::Pitch, 67});
    CHECK(Vocabulary::eventFor(tokens[3]) == TokenEvent {EventType::Program, 33});
    CHECK(Vocabulary::eventFor(tokens[6]) == TokenEvent {EventType::Tie, 0});
}
