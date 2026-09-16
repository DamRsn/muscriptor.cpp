#include "vectors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

namespace msl::test
{

namespace
{

    nlohmann::json readJson(const std::filesystem::path& inPath)
    {
        std::ifstream stream(inPath);

        if (!stream) {
            throw std::runtime_error(msl::format(
                "fixture not found at {}.\nRegenerate it with: uv run msl-tables && uv run msl-note-vectors",
                inPath.string()));
        }

        return nlohmann::json::parse(stream);
    }

    Note parseNote(const nlohmann::json& inJson)
    {
        Note note;
        note.onset = inJson.at("onset").get<double>();
        note.offset = inJson.at("offset").get<double>();
        note.pitch = inJson.at("pitch").get<int>();
        note.program = inJson.at("program").get<int>();
        note.is_drum = inJson.at("is_drum").get<bool>();
        return note;
    }

    std::vector<Note> parseNotes(const nlohmann::json& inJson)
    {
        std::vector<Note> notes;
        notes.reserve(inJson.size());
        std::transform(inJson.begin(), inJson.end(), std::back_inserter(notes), parseNote);
        return notes;
    }

    NoteAction parseAction(const nlohmann::json& inJson)
    {
        const std::string kind = inJson.at("kind").get<std::string>();
        NoteAction action;
        action.time = inJson.at("time").get<double>();
        action.pitch = inJson.at("pitch").get<int>();

        if (kind == "start") {
            action.kind = NoteActionKind::Start;
            action.program = inJson.at("program").get<int>();
        } else if (kind == "end") {
            action.kind = NoteActionKind::End;
            action.program = inJson.at("program").get<int>();
        } else if (kind == "drum") {
            action.kind = NoteActionKind::DrumHit;
            // Drum hits carry no program: the reference's _DrumHit has only a
            // pitch and a time, and the tracker never consults the register.
            action.program = 0;
        } else {
            throw std::runtime_error(msl::format("unknown note action kind '{}'", kind));
        }

        return action;
    }

    std::vector<NoteKey> parseKeys(const nlohmann::json& inJson)
    {
        std::vector<NoteKey> keys;
        keys.reserve(inJson.size());
        std::transform(inJson.begin(), inJson.end(), std::back_inserter(keys), [](const nlohmann::json& pair) {
            return NoteKey {pair.at(0).get<int>(), pair.at(1).get<int>()};
        });

        return keys;
    }

    /** Boundaries from seek times; the last chunk has no successor. */
    std::vector<ChunkBoundary> boundariesFor(const std::vector<double>& inSeekTimes)
    {
        std::vector<ChunkBoundary> out;
        out.reserve(inSeekTimes.size());

        for (std::size_t i = 0; i < inSeekTimes.size(); ++i) {
            ChunkBoundary boundary;
            boundary.seek_time = inSeekTimes[i];

            if (i + 1 < inSeekTimes.size()) {
                boundary.next_seek_time = inSeekTimes[i + 1];
            }

            out.push_back(boundary);
        }

        return out;
    }

} // namespace

std::vector<ChunkBoundary> NoteVector::boundaries() const
{
    return boundariesFor(seek_times);
}

Vectors::Vectors()
{
    const std::filesystem::path root = testdataRoot() / "vectors";

    const nlohmann::json tables = readJson(root / "tables.json");
    const nlohmann::json& vocab = tables.at("vocab");
    mNumTokens = vocab.at("num_tokens").get<std::int32_t>();
    mEosId = vocab.at("eos_id").get<std::int32_t>();

    for (const auto& [type, bounds]: vocab.at("ranges").items()) {
        mVocabRanges.emplace_back(type, std::pair {bounds.at(0).get<std::int32_t>(), bounds.at(1).get<std::int32_t>()});
    }

    for (const nlohmann::json& entry: vocab.at("spot_check")) {
        mSpotChecks.push_back(
            {entry.at(0).get<std::int32_t>(), entry.at(1).get<std::string>(), entry.at(2).get<std::int32_t>()});
    }

    const nlohmann::json& groups = tables.at("instrument_groups");
    mDrumProgram = groups.at("drum_program").get<int>();

    const nlohmann::json& group_map = groups.at("group_program_map");
    mNumGroups = static_cast<int>(group_map.size());
    mGroupPrograms.resize(static_cast<std::size_t>(mNumGroups));

    for (const auto& [gid, programs]: group_map.items()) {
        mGroupPrograms[static_cast<std::size_t>(std::stoi(gid))] = programs.get<std::vector<int>>();
    }

    for (const auto& [name, gid]: groups.at("names").items()) {
        mNamedGroups.emplace_back(name, gid.get<int>());
    }

    for (const auto& [program, name]: groups.at("program_to_name").items()) {
        mProgramNames.emplace_back(std::stoi(program), name.get<std::string>());
    }

    for (const nlohmann::json& entry: tables.at("forbidden_token_ids")) {
        mForbidden.push_back({entry.at("instruments").get<std::vector<std::string>>(),
                              entry.at("token_ids").get<std::vector<std::int32_t>>()});
    }

    for (const nlohmann::json& entry: tables.at("tie_section_token_ids")) {
        mTieSections.push_back(
            {parseKeys(entry.at("open_keys")), entry.at("token_ids").get<std::vector<std::int32_t>>()});
    }

    const nlohmann::json vectors = readJson(root / "note_vectors.json");

    for (const nlohmann::json& entry: vectors.at("vectors")) {
        NoteVector vector;
        vector.name = entry.at("name").get<std::string>();
        vector.description = entry.at("description").get<std::string>();
        vector.seek_times = entry.at("seek_times").get<std::vector<double>>();
        vector.chunk_tokens = entry.at("chunk_tokens").get<std::vector<std::vector<std::int32_t>>>();

        for (const nlohmann::json& keys: entry.at("open_keys_at_boundary")) {
            vector.open_keys_at_boundary.push_back(parseKeys(keys));
        }

        for (const nlohmann::json& action: entry.at("actions")) {
            vector.actions.push_back(parseAction(action));
        }

        vector.notes = parseNotes(entry.at("notes"));
        mNoteVectors.push_back(std::move(vector));
    }
}

const Vectors& Vectors::get()
{
    static const Vectors instance;
    return instance;
}

const NoteVector& Vectors::noteVector(const std::string& inName) const
{
    const auto it = std::find_if(
        mNoteVectors.begin(), mNoteVectors.end(), [&inName](const NoteVector& v) { return v.name == inName; });

    if (it == mNoteVectors.end()) {
        throw std::runtime_error(msl::format("no note vector named '{}'", inName));
    }

    return *it;
}

std::pair<std::int32_t, std::int32_t> Vectors::vocabRange(const std::string& inType) const
{
    const auto it = std::find_if(
        mVocabRanges.begin(), mVocabRanges.end(), [&inType](const auto& entry) { return entry.first == inType; });

    if (it == mVocabRanges.end()) {
        throw std::runtime_error(msl::format("no vocabulary range for '{}'", inType));
    }

    return it->second;
}

std::string Vectors::programName(int inProgram) const
{
    const auto it = std::find_if(mProgramNames.begin(), mProgramNames.end(), [inProgram](const auto& entry) {
        return entry.first == inProgram;
    });

    if (it == mProgramNames.end()) {
        throw std::runtime_error(msl::format("no reference name for program {}", inProgram));
    }

    return it->second;
}

const NoteReferences& requireReferences()
{
    const NoteReferences* refs = NoteReferences::get();

    if (refs == nullptr) {
        SKIP(msl::format("{} not found; generate it with: uv run msl-dump-refs --size {}",
                         (currentConfig().refsDir() / "notes.json").string(),
                         currentConfig().size));
    }

    return *refs;
}

std::vector<ChunkBoundary> VariantReference::boundaries() const
{
    return boundariesFor(seek_times);
}

NoteReferences::NoteReferences()
{
    const nlohmann::json notes = readJson(currentConfig().refsDir() / "notes.json");

    // fp32 is the comparison target throughout the suite; the dumper records
    // fp16 too and asserts the token streams agree, so there is nothing extra
    // to check here.
    for (const auto& [name, entry]: notes.at("fp32").items()) {
        VariantReference variant;
        variant.name = name;
        variant.prelude_forcing = entry.at("prelude_forcing").get<bool>();
        variant.instruments = entry.at("instruments").get<std::vector<std::string>>();
        variant.conditioning_rows = entry.at("conditioning_rows").get<std::vector<std::int32_t>>();
        variant.forbidden_token_ids = entry.at("forbidden_token_ids").get<std::vector<std::int32_t>>();
        variant.seek_times = entry.at("seek_times").get<std::vector<double>>();
        variant.tokens = entry.at("tokens").get<std::vector<std::vector<std::int32_t>>>();
        variant.prompts = entry.at("prompts").get<std::vector<std::vector<std::int32_t>>>();

        for (const nlohmann::json& keys: entry.at("open_keys_at_boundary")) {
            variant.open_keys_at_boundary.push_back(parseKeys(keys));
        }

        for (const nlohmann::json& action: entry.at("actions")) {
            variant.actions.push_back(parseAction(action));
        }

        variant.notes = parseNotes(entry.at("notes"));
        mVariants.push_back(std::move(variant));
    }
}

const NoteReferences* NoteReferences::get()
{
    if (!std::filesystem::exists(currentConfig().refsDir() / "notes.json")) {
        return nullptr;
    }

    static PerConfig<NoteReferences> instance;
    return &instance.get([] { return std::unique_ptr<NoteReferences>(new NoteReferences()); });
}

const VariantReference& NoteReferences::variant(const std::string& inName) const
{
    const auto it = std::find_if(
        mVariants.begin(), mVariants.end(), [&inName](const VariantReference& v) { return v.name == inName; });

    if (it == mVariants.end()) {
        throw std::runtime_error(msl::format("no reference variant named '{}'", inName));
    }

    return *it;
}

std::string describeNotes(const std::vector<Note>& inNotes)
{
    std::string out = msl::format("{} notes\n", inNotes.size());

    for (const Note& note: inNotes) {
        out += msl::format("  {} prog {}  pitch {}  {} -> {}\n",
                           note.is_drum ? "drum" : "note",
                           note.program,
                           note.pitch,
                           formatDouble("%.3f", note.onset),
                           formatDouble("%.3f", note.offset));
    }

    return out;
}

std::string describeActions(const std::vector<NoteAction>& inActions)
{
    static constexpr std::array<const char*, 3> KINDS {"start", "end", "drum"};
    std::string out = msl::format("{} actions\n", inActions.size());

    for (const NoteAction& action: inActions) {
        out += msl::format("  {} prog {}  pitch {}  {}\n",
                           KINDS[static_cast<std::size_t>(action.kind)],
                           action.program,
                           action.pitch,
                           formatDouble("%.3f", action.time));
    }

    return out;
}

} // namespace msl::test
