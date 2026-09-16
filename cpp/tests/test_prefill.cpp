// The prefill pass, checked stage by stage.
//
// This is the bisection ladder. Each stage is compared in the order the model
// computes it, so the first failing test names the first thing that went wrong
// instead of leaving a wrong logit to be traced backwards by hand. Layer 0 is
// opened up completely (norm, attention context, projection, both residuals,
// the FFN either side of the GELU) because a block that is wrong is usually
// wrong in one specific place; the remaining layers are compared at their
// outputs, which is enough to bracket a divergence that only appears with
// depth.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace msl;
using namespace msl::test;

namespace
{

void check_stage(const std::string& name)
{
    const Reference& ref = Reference::fp32();
    const Tolerance tol = ref.tolerance(name);
    const Comparison comp = compare(tracedPrefill().trace.read(name), ref.tensor(name));
    INFO(describe(name, comp, tol));
    recordParity(name, comp, tol);
    CHECK_FALSE(comp.nonfinite_mismatch);
    CHECK(comp.max_abs <= tol.atol);
    CHECK(comp.cosine >= tol.cosine_min);
}

} // namespace

TEST_CASE("token embedding matches the reference", "[prefill]")
{
    check_stage("pre.tok_embed");
}

TEST_CASE("conditioning prefix is assembled in the right order", "[prefill]")
{
    // ConditioningProvider iterates {instrument_group, dataset_name, self_wav}
    // and each condition is prepended in front of the previous one, so the
    // sequence ends up as [mel, dataset_name, instrument_group, tokens] -- the
    // reverse of the iteration order. Swapping the two class embeddings is
    // invisible in every shape check and changes every logit.
    check_stage("pre.prefix");
}

TEST_CASE("positions are added to the prefix", "[prefill]")
{
    check_stage("pre.layer_in");
}

TEST_CASE("layer 0 attention internals match the reference", "[prefill][layer0]")
{
    check_stage("blk.0.norm1");
    check_stage("blk.0.attn_ctx");
    check_stage("blk.0.attn_out");
    check_stage("blk.0.res1");
}

TEST_CASE("layer 0 feed-forward internals match the reference", "[prefill][layer0]")
{
    check_stage("blk.0.norm2");
    check_stage("blk.0.ffn_pre_gelu");
    // F.gelu is the exact erf formulation. ggml_gelu is the tanh approximation
    // and differs by ~1e-3, which survives to the logits, so this stage is
    // really a test that the port picked ggml_gelu_erf.
    check_stage("blk.0.ffn_gelu");
    check_stage("blk.0.ffn_out");
}

TEST_CASE("every layer output matches the reference", "[prefill][layers]")
{
    const Reference& ref = Reference::fp32();

    for (int il = 0; il < ref.n_layer(); ++il) {
        const std::string name = msl::format("blk.{}.out", il);
        INFO("layer " << il);
        check_stage(name);
    }
}

TEST_CASE("output norm matches the reference", "[prefill]")
{
    check_stage("post.out_norm");
}

TEST_CASE("prefill logits and first token match the reference", "[prefill][logits]")
{
    const Reference& ref = Reference::fp32();
    const TracedPrefill& prefill = tracedPrefill();
    const std::vector<float>& logits = prefill.logits;

    const Tolerance tol = ref.tolerance("post.logits");
    const Comparison comp = compare(logits, ref.tensor("post.logits"));
    INFO(describe("post.logits", comp, tol));
    recordParity("post.logits", comp, tol);
    CHECK_FALSE(comp.nonfinite_mismatch);
    CHECK(comp.max_abs <= tol.atol);
    CHECK(comp.cosine >= tol.cosine_min);

    // The only thing that actually propagates is the argmax, so it is checked
    // as a discrete outcome rather than inferred from the tolerance passing.
    const auto best = std::max_element(logits.begin(), logits.end());
    CHECK(static_cast<std::int32_t>(std::distance(logits.begin(), best)) == ref.decode_argmax(0));
    CHECK(prefill.n_past == ref.prepend_length() + 1);
}
