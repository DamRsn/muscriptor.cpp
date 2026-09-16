// Whole chunks, token for token.
//
// Three independent 5-second chunks, greedy, decoded to EOS, compared token for
// token against the reference. Independent means prelude forcing is off, so no
// chunk depends on the previous chunk's open notes.
//
// Chunks 1 and 2 are the reason this is not a one-chunk test: they only pass if
// reset() genuinely clears the KV cache and the position counter, and a stale
// cache is the kind of bug that leaves chunk 0 perfect.
//
// The comparison reads the shared teacher-forced runs: a chunk whose every
// position picks the reference's token is the chunk greedy decoding produces.
// generate() itself is driven on a whole chunk by test_stft.cpp and by the
// Transcriber, and its stopping rule is pinned below.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace msl;
using namespace msl::test;

namespace
{

void requireFollowsReference(int inChunk)
{
    const Reference& ref = Reference::fp32();
    const ForcedChunk& run = plainChunk(inChunk);
    const std::vector<std::int32_t>& want = ref.tokens(inChunk);

    REQUIRE(want.back() == ref.eos_id());
    REQUIRE(run.steps.size() == want.size());

    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("token " << i << ", margin " << run.steps[i].margin);
        REQUIRE(run.steps[i].want == want[i]);
        REQUIRE(run.steps[i].argmax == want[i]);
    }
}

} // namespace

TEST_CASE("chunk 0 follows the reference token stream to EOS", "[generate]")
{
    requireFollowsReference(0);
}

TEST_CASE("a budget that runs out costs no extra forward pass", "[generate]")
{
    // A chunk that never emits EOS is a normal outcome, not an error, so the
    // exhausted path is worth being exact about: the last token the budget
    // allows is emitted, and the forward pass that would compute what follows
    // it is not run, because nothing would read it.
    //
    // Counted in KV positions rather than in wall time, which is the only way
    // to state it that does not depend on the machine.
    const Reference& ref = Reference::fp32();
    Model& model = shared_model();

    constexpr int BUDGET = 5;
    REQUIRE(ref.tokens(0).size() > static_cast<std::size_t>(BUDGET)); // or it would stop on EOS instead

    const std::vector<std::int32_t> got =
        model.generate(ref.tensor("cond.embed"), ref.mel_frames(), BUDGET, ref.eos_id());

    REQUIRE(got.size() == BUDGET);
    CHECK(got.back() != ref.eos_id());

    for (int i = 0; i < BUDGET; ++i) {
        INFO("token " << i);
        CHECK(got[static_cast<std::size_t>(i)] == ref.tokens(0)[static_cast<std::size_t>(i)]);
    }

    // mel frames + one dataset row + the instrument rows, then the initial
    // token, then one decode per token *except* the last.
    const int prefix = ref.mel_frames() + 1 + static_cast<int>(model.instrumentRows().size());
    CHECK(model.nPast() == prefix + 1 + BUDGET - 1);
}

TEST_CASE("every chunk follows the reference token stream to EOS", "[generate][chunks][slow]")
{
    const Reference& ref = Reference::fp32();

    for (int chunk = 0; chunk < ref.num_chunks(); ++chunk) {
        INFO("chunk " << chunk);
        requireFollowsReference(chunk);
    }
}
