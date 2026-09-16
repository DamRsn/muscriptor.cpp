// Prelude forcing through the model, chunk by chunk.
//
// The prompt itself is already pinned without any inference in
// test_tokens_to_notes.cpp -- openKeys() at each boundary, and what
// tieSectionTokenIds encodes it to. What is left to check here is the part that
// needs the transformer: that feeding that prompt to prefill in one
// square-causal pass reproduces the reference's token stream, and that the
// prompt comes back out of generate() so the decode state machine sees it.
//
// This is the default configuration upstream, so it is the one that matters
// most for output quality: it is specifically what stops a sustained note from
// re-entering under the wrong instrument at every chunk seam.

#include "inference.hpp"

#include "vocabulary.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>

using namespace msl;
using namespace msl::test;

TEST_CASE("generate returns its prompt alongside what it generated", "[prelude]")
{
    // Not cosmetic. The prompt tokens are what leave the tie prologue, so a
    // generate() that swallowed them would leave the tracker reading the whole
    // chunk body in the prologue's grammar -- every note declared sustained and
    // nothing ever onset.
    const Reference& ref = Reference::fp32();
    Model& model = shared_model();

    const std::array<NoteKey, 2> open {{{0, 60}, {0, 64}}};
    const std::vector<std::int32_t> prompt = Vocabulary::tieSectionTokenIds(open);

    const std::vector<std::int32_t> got = model.generate(
        ref.tensor("cond.embed"), ref.mel_frames(), static_cast<int>(prompt.size()) + 16, ref.eos_id(), prompt);

    REQUIRE(got.size() > prompt.size());
    CHECK(std::vector<std::int32_t>(got.begin(), got.begin() + static_cast<std::ptrdiff_t>(prompt.size())) == prompt);

    // The prompt rides in the prefill, so the position counter advances by the
    // prefix plus the initial token plus the prompt -- not by the prompt twice.
    // Every returned token is then fed back except the last one, which never is:
    // either it is the EOS that ended the loop, or the budget ran out and the
    // pass would have produced logits nothing reads.
    const int fed = static_cast<int>(got.size()) - 1;
    CHECK(model.nPast() == ref.prepend_length() + 1 + fed);
}

TEST_CASE("a forced chunk follows the reference step by step", "[prelude][slow]")
{
    // Driven exactly as the Transcriber drives it: the tracker walks the chunks
    // before each boundary, its open keys are encoded, and the result is forced
    // into the prefill.
    //
    // The forced prologue is compared exactly -- it is integer logic and there
    // is nothing to be approximate about. The decode that follows is compared
    // per step, teacher-forced, with near-ties classified against the noise
    // floor the dumper measured; see test_instrument_conditioning.cpp.
    const VariantReference& variant = requireReferences().variant("prelude");
    REQUIRE(variant.prelude_forcing);

    int near_ties = 0;

    for (std::size_t chunk = 0; chunk < variant.tokens.size(); ++chunk) {
        INFO("chunk " << chunk);
        const ForcedChunk& run = variantChunk("prelude", static_cast<int>(chunk));

        REQUIRE(run.prompt == variant.prompts[chunk]);
        near_ties += requireOnlyNearTies(run);
    }

    if (near_ties > 0) {
        WARN(msl::format("{} argmax near-tie(s) inside the measured noise floor; "
                         "re-run with --weight-dtype f32, where there should be none",
                         near_ties));
    }
}
