#pragma once

#include "muscriptor/note.hpp"

#include "open_note_tracker.hpp"
#include "run_config.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace msl::test
{

/**
 * One hand-authored token-stream case from `testdata/vectors/note_vectors.json`,
 * decoded by the reference.
 */
struct NoteVector {
    std::string name;
    std::string description;
    std::vector<double> seek_times;
    std::vector<std::vector<std::int32_t>> chunk_tokens;
    // The open keys prelude forcing would read, one entry per boundary.
    std::vector<std::vector<NoteKey>> open_keys_at_boundary;

    std::vector<NoteAction> actions;
    std::vector<Note> notes;

    /** Boundaries reconstructed from `seek_times`; the last one has no successor. */
    std::vector<ChunkBoundary> boundaries() const;
};

/** One instrument selection and the token ids it forbids. */
struct ForbiddenCase {
    std::vector<std::string> instruments;
    std::vector<std::int32_t> token_ids;
};

/** One open-note set and the tie prologue it encodes to. */
struct TieSectionCase {
    std::vector<NoteKey> open_keys;
    std::vector<std::int32_t> token_ids;
};

/** One id at a vocabulary range boundary, and what it decodes to. */
struct VocabSpotCheck {
    std::int32_t token_id = 0;
    std::string type;
    std::int32_t value = 0;
};

/**
 * The committed, weight-free fixtures.
 *
 * Deliberately separate from `Reference`, which throws when the converted
 * weights are missing -- that strictness is a feature there and would make the
 * pure integer tests unrunnable on a bare checkout. Nothing here reads a GGUF
 * or needs a model.
 */
class Vectors
{
public:
    /** Loaded once and shared; the JSON is small but every test case wants it. */
    static const Vectors& get();

    const std::vector<NoteVector>& noteVectors() const { return mNoteVectors; }
    const NoteVector& noteVector(const std::string& inName) const;

    // --- tables.json ---
    std::int32_t numTokens() const { return mNumTokens; }
    std::int32_t eosId() const { return mEosId; }
    int numGroups() const { return mNumGroups; }
    int drumProgram() const { return mDrumProgram; }

    /** Range of token ids for an event type name, as [first, last]. */
    std::pair<std::int32_t, std::int32_t> vocabRange(const std::string& inType) const;
    const std::vector<VocabSpotCheck>& vocabSpotChecks() const { return mSpotChecks; }

    /** Group id -> its programs, exactly as the reference builds them. */
    const std::vector<std::vector<int>>& groupPrograms() const { return mGroupPrograms; }

    /** Name -> group id, for the named groups only. */
    const std::vector<std::pair<std::string, int>>& namedGroups() const { return mNamedGroups; }

    /** Decoded program -> the name the reference gives it. */
    std::string programName(int inProgram) const;

    const std::vector<ForbiddenCase>& forbiddenCases() const { return mForbidden; }
    const std::vector<TieSectionCase>& tieSectionCases() const { return mTieSections; }

private:
    Vectors();

    std::vector<NoteVector> mNoteVectors;

    std::int32_t mNumTokens = 0;
    std::int32_t mEosId = 0;
    int mNumGroups = 0;
    int mDrumProgram = 0;
    std::vector<std::pair<std::string, std::pair<std::int32_t, std::int32_t>>> mVocabRanges;
    std::vector<VocabSpotCheck> mSpotChecks;
    std::vector<std::vector<int>> mGroupPrograms;
    std::vector<std::pair<std::string, int>> mNamedGroups;
    std::vector<std::pair<int, std::string>> mProgramNames;
    std::vector<ForbiddenCase> mForbidden;
    std::vector<TieSectionCase> mTieSections;
};

/** One decode configuration's reference output, from `testdata/refs/<size>/notes.json`. */
struct VariantReference {
    std::string name;
    bool prelude_forcing = false;
    std::vector<std::string> instruments;
    std::vector<std::int32_t> conditioning_rows;
    std::vector<std::int32_t> forbidden_token_ids;
    std::vector<double> seek_times;

    // Per chunk, EOS already stripped and the teacher-forced prologue included
    // -- the same thing Model::generate returns.
    std::vector<std::vector<std::int32_t>> tokens;
    // Per chunk, the forced prologue alone. Empty for chunk 0 and when forcing
    // is off.
    std::vector<std::vector<std::int32_t>> prompts;
    std::vector<std::vector<NoteKey>> open_keys_at_boundary;

    std::vector<NoteAction> actions;
    std::vector<Note> notes;

    std::vector<ChunkBoundary> boundaries() const;
};

/**
 * The generated note-level references.
 *
 * Separate from `Reference` because nothing here needs the converted weights --
 * these are real model token streams decoded by the reference, so the whole
 * integer stack can be validated against them with no inference at all.
 */
class NoteReferences
{
public:
    /** Returns nullptr when `notes.json` has not been dumped for the current size. */
    static const NoteReferences* get();

    const VariantReference& variant(const std::string& inName) const;
    const std::vector<VariantReference>& variants() const { return mVariants; }

private:
    NoteReferences();

    std::vector<VariantReference> mVariants;
};

/** @return The note-level references; SKIPs the test when they were never dumped. */
const NoteReferences& requireReferences();

/** @return `inNotes` formatted one per line, for a failing test's INFO block. */
std::string describeNotes(const std::vector<Note>& inNotes);

/** @return `inActions` formatted one per line. */
std::string describeActions(const std::vector<NoteAction>& inActions);

} // namespace msl::test
