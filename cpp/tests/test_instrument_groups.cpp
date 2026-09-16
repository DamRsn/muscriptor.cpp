// The instrument group table, asserted against the reference's own.
//
// cpp/src/instrument_groups.inc is generated from testdata/vectors/tables.json
// rather than written, because upstream derives its singleton groups by
// iterating a Python set -- so the group-id to program mapping above 35 is an
// artifact of CPython's ordering. That makes this file the thing that catches
// the generated table going stale after an upstream bump, which is the only
// realistic way for it to be wrong.
//
// Runs with no weights and no reference dump -- tagged [pure].

#include "vectors.hpp"

#include "muscriptor/error.hpp"

#include "instrument_groups.hpp"
#include "vocabulary.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <set>

using namespace msl;
using namespace msl::test;

TEST_CASE("the frozen group table matches the reference", "[pure][instruments]")
{
    const Vectors& vectors = Vectors::get();

    REQUIRE(InstrumentGroups::numGroups() == vectors.numGroups());
    REQUIRE(DRUM_PROGRAM == vectors.drumProgram());

    for (int gid = 0; gid < vectors.numGroups(); ++gid) {
        INFO("group " << gid);
        const std::vector<int>& programs = vectors.groupPrograms()[static_cast<std::size_t>(gid)];
        REQUIRE_FALSE(programs.empty());
        // The model only ever emits a group's first program, so that is the
        // only entry the port needs -- and the only one it stores.
        CHECK(InstrumentGroups::representativeProgram(gid) == programs.front());
    }
}

TEST_CASE("every named group has the reference's id and name", "[pure][instruments]")
{
    const Vectors& vectors = Vectors::get();

    for (const auto& [name, gid]: vectors.namedGroups()) {
        INFO("group " << name << " (" << gid << ")");
        const std::optional<InstrumentGroup> group = InstrumentGroups::groupForName(name);
        REQUIRE(group.has_value());
        CHECK(InstrumentGroups::groupIdFor(*group) == gid);
        CHECK(InstrumentGroups::nameForGroupId(gid) == name);
        CHECK(instrumentName(*group) == name);
    }

    CHECK(InstrumentGroups::groupForName("not_an_instrument") == std::nullopt);
}

TEST_CASE("a decoded program resolves to the name the reference gives it", "[pure][instruments]")
{
    const Vectors& vectors = Vectors::get();

    // Programs 0-129 are what a `program` token can carry; DRUM_PROGRAM arrives
    // only from a drum hit but has to name itself too.
    for (int program = 0; program < Vocabulary::PROGRAM_COUNT; ++program) {
        INFO("program " << program);
        CHECK(instrumentLabel(program) == vectors.programName(program));
    }

    CHECK(instrumentLabel(DRUM_PROGRAM) == vectors.programName(DRUM_PROGRAM));
}

TEST_CASE("program 96 decodes as drums, on purpose", "[pure][instruments]")
{
    // Group 36 is both the singleton group for GM program 96 (Sound FX "Rain")
    // and the group the reference names "drums", so a decoded 96 is a drum, as
    // in the reference.
    REQUIRE(InstrumentGroups::representativeProgram(36) == 96);
    CHECK(instrumentGroupFor(96) == InstrumentGroup::Drums);
    CHECK(instrumentLabel(96) == "drums");
}

TEST_CASE("unnamed singleton groups keep their program instead of a name", "[pure][instruments]")
{
    // 97 through 127 (bar 96) are the leftovers upstream gave singleton groups
    // and never named. Flattening them onto one "unknown" would lose which
    // instrument the model actually meant, which is why Note carries the raw
    // program and this returns nothing.
    CHECK(instrumentGroupFor(97) == std::nullopt);
    CHECK(instrumentLabel(97) == "program_97");
    CHECK(instrumentGroupFor(127) == std::nullopt);
    CHECK(instrumentLabel(127) == "program_127");
}

TEST_CASE("a program that is not a representative has no group", "[pure][instruments]")
{
    // Program 1 sits inside group 0, whose representative is 0. The model never
    // emits it, and if it somehow did the reference would call it program_1.
    CHECK(instrumentGroupFor(1) == std::nullopt);
    CHECK(instrumentLabel(1) == "program_1");
}

TEST_CASE("programFor inverts the group lookup", "[pure][instruments]")
{
    for (const auto& [name, gid]: Vectors::get().namedGroups()) {
        INFO("group " << name);
        const InstrumentGroup group = *InstrumentGroups::groupForName(name);

        if (group == InstrumentGroup::Drums) {
            // Drums are not a program group: hits carry DRUM_PROGRAM, not the
            // group's nominal representative of 96.
            CHECK(programFor(group) == DRUM_PROGRAM);
        } else {
            CHECK(programFor(group) == InstrumentGroups::representativeProgram(gid));
            CHECK(instrumentGroupFor(programFor(group)) == group);
        }
    }
}

TEST_CASE("conditioning rows are the group id plus two", "[pure][instruments]")
{
    // tokenize adds one and forward adds one again. The null class is row 1,
    // which is what an empty selection has to produce.
    CHECK(InstrumentGroups::conditioningRow(InstrumentGroup::AcousticPiano) == 2);
    CHECK(InstrumentGroups::conditioningRow(InstrumentGroup::ElectricBass) == 10);
    CHECK(InstrumentGroups::conditioningRow(InstrumentGroup::Drums) == 38);

    CHECK(InstrumentGroups::conditioningRows({}) == std::vector<std::int32_t> {1});

    // Order follows the caller's, not a sort: the conditioning string upstream
    // builds is in the order the names were given.
    const std::array<InstrumentGroup, 2> selection {InstrumentGroup::Drums, InstrumentGroup::ElectricBass};
    CHECK(InstrumentGroups::conditioningRows(selection) == std::vector<std::int32_t> {38, 10});
}

TEST_CASE("forbidden token ids match the reference for every selection", "[pure][instruments]")
{
    const Vectors& vectors = Vectors::get();

    for (const ForbiddenCase& expected: vectors.forbiddenCases()) {
        if (expected.instruments.empty()) {
            continue; // covered by its own case below
        }

        std::vector<InstrumentGroup> groups;
        for (const std::string& name: expected.instruments) {
            const std::optional<InstrumentGroup> group = InstrumentGroups::groupForName(name);
            REQUIRE(group.has_value());
            groups.push_back(*group);
        }

        INFO("selection: " << expected.instruments.size() << " instrument(s)");
        std::vector<std::int32_t> got = InstrumentGroups::forbiddenTokenIds(groups);
        std::vector<std::int32_t> want = expected.token_ids;
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        CHECK(got == want);
    }
}

TEST_CASE("an empty selection is rejected rather than forbidding everything", "[pure][instruments]")
{
    // The reference's forbidden_token_ids([]) forbids all 130 programs and all
    // 128 drums, and upstream never calls it that way -- transcribe() guards on
    // the empty selection. Silently returning that here would turn "no filter"
    // into "no instruments", so it throws and Transcriber does the guarding.
    const Vectors& vectors = Vectors::get();
    const auto it = std::find_if(vectors.forbiddenCases().begin(),
                                 vectors.forbiddenCases().end(),
                                 [](const ForbiddenCase& c) { return c.instruments.empty(); });

    REQUIRE(it != vectors.forbiddenCases().end());
    CHECK(it->token_ids.size() == static_cast<std::size_t>(Vocabulary::PROGRAM_COUNT + Vocabulary::DRUM_COUNT));

    CHECK_THROWS_AS(InstrumentGroups::forbiddenTokenIds({}), Exception);
}

TEST_CASE("the forbidden mask never touches timing, pitch or the specials", "[pure][instruments]")
{
    const std::array<InstrumentGroup, 1> selection {InstrumentGroup::ElectricBass};
    const std::vector<std::int32_t> forbidden = InstrumentGroups::forbiddenTokenIds(selection);
    const std::set<std::int32_t> masked(forbidden.begin(), forbidden.end());

    for (std::int32_t id = 0; id < Vocabulary::NUM_TOKENS; ++id) {
        const EventType type = Vocabulary::eventFor(id).type;
        const bool maskable = type == EventType::Program || type == EventType::Drum;

        if (!maskable) {
            INFO("token " << id << " is not a program or drum token");
            REQUIRE_FALSE(masked.contains(id));
        }
    }

    // Exactly one program survives -- electric bass's representative, 33 -- and
    // every drum is masked because drums were not selected.
    CHECK(masked.size() == static_cast<std::size_t>(Vocabulary::PROGRAM_COUNT - 1 + Vocabulary::DRUM_COUNT));
    CHECK_FALSE(masked.contains(Vocabulary::tokenFor(EventType::Program, 33)));
}

TEST_CASE("selecting drums allows drum tokens and no program", "[pure][instruments]")
{
    const std::array<InstrumentGroup, 1> selection {InstrumentGroup::Drums};
    const std::vector<std::int32_t> forbidden = InstrumentGroups::forbiddenTokenIds(selection);
    const std::set<std::int32_t> masked(forbidden.begin(), forbidden.end());

    // Drums contribute no allowed program, not even group 36's nominal
    // representative of 96 -- the reference skips them in that loop entirely.
    CHECK(masked.size() == static_cast<std::size_t>(Vocabulary::PROGRAM_COUNT));
    CHECK(masked.contains(Vocabulary::tokenFor(EventType::Program, 96)));

    for (std::int32_t value = 0; value < Vocabulary::DRUM_COUNT; ++value) {
        REQUIRE_FALSE(masked.contains(Vocabulary::tokenFor(EventType::Drum, value)));
    }
}

TEST_CASE("every named group is enumerable for a UI", "[pure][instruments]")
{
    // The enumerators are the reference's group ids, so they run 0-33 and then
    // jump to 36. A picker cannot just count to N, which is what this exists
    // to spare every caller.
    const std::span<const InstrumentGroup> all = allInstrumentGroups();

    CHECK(all.size() == Vectors::get().namedGroups().size());
    CHECK(all.size() == static_cast<std::size_t>(InstrumentGroups::MAX_SELECTABLE));

    for (const InstrumentGroup group: all) {
        INFO("group " << static_cast<int>(group));
        REQUIRE_FALSE(instrumentName(group).empty());
        REQUIRE(InstrumentGroups::groupForName(instrumentName(group)) == group);
    }

    CHECK(std::find(all.begin(), all.end(), InstrumentGroup::Drums) != all.end());
    CHECK(std::find(all.begin(), all.end(), InstrumentGroup::AcousticPiano) != all.end());
}
