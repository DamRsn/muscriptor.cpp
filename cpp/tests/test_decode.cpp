// The KV-cached decode loop.
//
// The decode step is where a port usually breaks: the cache write offset, the
// causal mask alignment and the position index all have to advance together,
// and each of them fails in a way that still produces plausible logits. Two
// details from the reference matter here:
//
//   * `offset` advances by the whole prefill length (prefix + tokens), so the
//     first decoded token sits at position prepend_length + 1;
//   * causality is bottom-right aligned. PyTorch's is_causal=True is top-left
//     aligned, which the reference sidesteps by special-casing T_q == 1; the
//     port builds an explicit mask that covers both shapes.
//
// Each step is compared against the reference's own logits, so a drift that
// only appears after several steps is caught at the step it starts. Both cases
// read chunk 0's shared teacher-forced run.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace msl;
using namespace msl::test;

TEST_CASE("decode steps match the reference logits", "[decode]")
{
    const Reference& ref = Reference::fp32();
    const ForcedChunk& run = plainChunk(0);
    REQUIRE(run.decode_logits.size() == static_cast<std::size_t>(ref.max_decode_steps()));

    for (int step = 1; step <= ref.max_decode_steps(); ++step) {
        const auto index = static_cast<std::size_t>(step - 1);
        const std::vector<float>& logits = run.decode_logits[index];

        const std::string name = msl::format("dec.step{}.logits", step);
        const Tolerance tol = ref.tolerance(name);
        const Comparison comp = compare(logits, ref.tensor(name));
        INFO("decode step " << step << "\n" << describe(name, comp, tol));
        recordParity(name, comp, tol);
        CHECK_FALSE(comp.nonfinite_mismatch);
        CHECK(comp.max_abs <= tol.atol);
        CHECK(comp.cosine >= tol.cosine_min);

        const auto best = std::max_element(logits.begin(), logits.end());
        CHECK(static_cast<std::int32_t>(std::distance(logits.begin(), best)) == ref.decode_argmax(step));

        CHECK(run.n_past_after_decode[index] == ref.prepend_length() + 1 + step);
    }
}

TEST_CASE("free-running decode reproduces the reference tokens", "[decode]")
{
    // Every position picks the reference's token, so greedy decoding fed its own
    // argmax walks exactly this path. Passing the previous case and failing this
    // one would mean the logits are close and the choice between them is not.
    const Reference& ref = Reference::fp32();
    const ForcedChunk& run = plainChunk(0);
    REQUIRE(run.steps.size() > static_cast<std::size_t>(ref.max_decode_steps()));

    for (std::size_t step = 0; step < static_cast<std::size_t>(ref.max_decode_steps()); ++step) {
        INFO("decode step " << step);
        REQUIRE(run.steps[step].argmax == run.steps[step].want);
    }
}
