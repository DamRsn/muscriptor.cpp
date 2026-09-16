// Instrument selection: the advisory conditioning prefix and the hard mask.
//
// The two mechanisms are independent and the reference uses both. The soft one
// changes the prefix *length* -- one position per selected group, against the
// single null-class row of the unconditional path. The hard one is a logit mask
// and cannot change a length, but it can silently do nothing.
//
// pre.prefix pins the prefix's length and its order in one tensor.
//
// Two selections are dumped: `bass`, one row, and `band`, five rows including
// drums.

#include "inference.hpp"

#include "instrument_groups.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <set>

using namespace msl;
using namespace msl::test;

namespace
{

constexpr std::array<const char*, 2> SELECTIONS {"bass", "band"};

} // namespace

TEST_CASE("the unconditional prefix is a single null-class row", "[instruments]")
{
    // The default has to stay exactly one null-class row, or every number in
    // the bisection ladder moves.
    Model& model = shared_model();
    REQUIRE(model.instrumentRows().size() == 1);
    CHECK(model.instrumentRows()[0] == InstrumentGroups::NULL_CONDITIONING_ROW);
    CHECK_FALSE(model.hasForbiddenTokens());
}

TEST_CASE("a selection produces the reference's conditioning rows", "[instruments]")
{
    const Reference& ref = Reference::fp32();
    Model& model = shared_model();

    for (const std::string name: SELECTIONS) {
        const VariantReference& variant = requireReferences().variant(name);
        const std::string tensor_name = msl::format("cond.instrument_group.{}", name);

        INFO("variant " << name);
        REQUIRE(ref.has(tensor_name));

        const std::vector<InstrumentGroup> groups = groupsFor(variant);
        const ScopedSelection selection(model, groups);

        // The rows the port would read, against the ones the dumper recorded.
        const std::vector<std::int32_t> rows(model.instrumentRows().begin(), model.instrumentRows().end());
        CHECK(rows == variant.conditioning_rows);

        // And the embedding they select. Compared by element count rather than
        // by shape: GGUF trims trailing 1s from ne, so a single row comes back
        // as [dim] and five as [dim, 5].
        const std::size_t expected = variant.conditioning_rows.size() * static_cast<std::size_t>(model.hparams().dim);
        CHECK(ref.tensor(tensor_name).size() == expected);
    }
}

TEST_CASE("the conditioned prefix matches the reference in length and order", "[instruments]")
{
    const Reference& ref = Reference::fp32();

    for (const std::string name: SELECTIONS) {
        const VariantReference& variant = requireReferences().variant(name);
        const std::string prefix_name = msl::format("pre.prefix.{}", name);

        INFO("variant " << name);
        REQUIRE(ref.has(prefix_name));

        const TracedPrefill& prefill = tracedPrefill(name);
        const std::vector<float>& got = prefill.trace.read("pre.prefix");
        const std::vector<float>& want = ref.tensor(prefix_name);

        // mel + one dataset row + one row per instrument, then the initial token.
        const int expected_prepend = ref.mel_frames() + 1 + static_cast<int>(variant.conditioning_rows.size());
        CHECK(prefill.n_past == expected_prepend + 1);

        REQUIRE(got.size() == want.size());
        const Tolerance tol = ref.tolerance(prefix_name);
        const Comparison comp = compare(got, want);
        INFO(describe(prefix_name, comp, tol));
        recordParity(prefix_name, comp, tol);
        CHECK_FALSE(comp.nonfinite_mismatch);
        CHECK(comp.max_abs <= tol.atol);
        CHECK(comp.cosine >= tol.cosine_min);
    }
}

TEST_CASE("the forbidden mask reaches the logits", "[instruments]")
{
    const VariantReference& variant = requireReferences().variant("bass");
    const std::vector<float>& logits = tracedPrefill("bass").logits;

    // Applied by prefill, not only inside the sampling loop -- _compute_logits
    // masks on every forward pass, and a hand-driven caller has to see the same
    // logits the sampler would.
    for (const std::int32_t id: variant.forbidden_token_ids) {
        INFO("forbidden token " << id);
        REQUIRE(std::isinf(logits[static_cast<std::size_t>(id)]));
        REQUIRE(logits[static_cast<std::size_t>(id)] < 0.0f);
    }

    // And nothing else was masked beyond the reserved ids the model already
    // forces, so the mask is a filter rather than a blanket.
    const std::set<std::int32_t> forbidden(variant.forbidden_token_ids.begin(), variant.forbidden_token_ids.end());

    for (std::int32_t id = 0; id < shared_model().hparams().logit_mask_start; ++id) {
        if (!forbidden.contains(id)) {
            INFO("token " << id << " should not be masked");
            REQUIRE(std::isfinite(logits[static_cast<std::size_t>(id)]));
        }
    }
}

TEST_CASE("conditioned decoding follows the reference step by step", "[instruments][slow]")
{
    // Teacher-forced, so a disagreement is isolated to the step it starts at
    // instead of cascading into a different stream.
    //
    // The bar is the argmax. A disagreement whose top-two margin is inside the
    // reference's measured fp32/fp16 logit gap is counted as a near-tie;
    // anything outside it fails.
    int near_ties = 0;

    for (const std::string name: SELECTIONS) {
        const VariantReference& variant = requireReferences().variant(name);
        INFO("variant " << name);

        for (int chunk = 0; chunk < static_cast<int>(variant.tokens.size()); ++chunk) {
            INFO("chunk " << chunk);
            near_ties += requireOnlyNearTies(variantChunk(name, chunk));
        }
    }

    if (near_ties > 0) {
        WARN(msl::format("{} argmax near-tie(s) inside the measured noise floor; "
                         "re-run with --weight-dtype f32, where there should be none",
                         near_ties));
    }
}
