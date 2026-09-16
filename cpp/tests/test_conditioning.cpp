// The conditioning front-end.
//
// Entry point is the STFT magnitude spectrum dumped from the reference, so
// these stages are checked independently of the port's own STFT
// (test_stft.cpp).
//
// This whole path stays fp32 in the reference even when the transformer is
// fp16 -- load_model calls condition_provider.float() because the log of a
// quiet mel bin underflows in half precision -- so the tolerances here are the
// tightest in the suite and any real slack points at a genuine bug.

#include "inference.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace msl;
using namespace msl::test;

namespace
{

/** The conditioning graph over chunk 0, every stage traced, run once per configuration. */
struct ConditioningRun {
    Trace trace;
    std::vector<float> embedding;
};

const ConditioningRun& conditioningRun()
{
    static PerConfig<ConditioningRun> run;
    return run.get([] {
        const Reference& ref = Reference::fp32();
        auto out = std::make_unique<ConditioningRun>();
        out->embedding = shared_model().encodeConditioning(
            ref.tensor("in.spectrum"), ref.mel_frames(), ref.segment_samples(), &out->trace);
        return out;
    });
}

void check_stage(const Trace& trace, const std::string& name)
{
    const Reference& ref = Reference::fp32();
    const Tolerance tol = ref.tolerance(name);
    const Comparison comp = compare(trace.read(name), ref.tensor(name));
    INFO(describe(name, comp, tol));
    recordParity(name, comp, tol);
    CHECK_FALSE(comp.nonfinite_mismatch);
    CHECK(comp.max_abs <= tol.atol);
    CHECK(comp.cosine >= tol.cosine_min);
}

} // namespace

TEST_CASE("mel filterbank matches the reference", "[conditioning]")
{
    check_stage(conditioningRun().trace, "cond.mel");
}

TEST_CASE("log-mel matches the reference", "[conditioning]")
{
    check_stage(conditioningRun().trace, "cond.logmel");
}

TEST_CASE("mel projection matches the reference", "[conditioning]")
{
    check_stage(conditioningRun().trace, "cond.proj");
}

TEST_CASE("masked conditioning embedding matches the reference", "[conditioning]")
{
    const ConditioningRun& run = conditioningRun();
    check_stage(run.trace, "cond.embed");

    // The mask is derived from the waveform length, not the frame count: a
    // centre-padded STFT always emits one frame past length/hop, and that last
    // frame must come out exactly zero rather than merely small.
    const Reference& ref = Reference::fp32();
    const Hparams& hp = shared_model().hparams();
    const int last_frame = ref.mel_frames() - 1;

    for (int i = 0; i < hp.dim; ++i) {
        REQUIRE(run.embedding[static_cast<std::size_t>(last_frame) * hp.dim + i] == 0.0f);
    }
}

TEST_CASE("class embeddings resolve to the null class", "[conditioning]")
{
    // Both ClassConditioners land on row 1, not row 0: tokenize(None) gives
    // 1 + (-1) == 0 and forward() embeds inputs + 1. Reading row 0 would still
    // produce plausible-looking numbers, so this is checked directly.
    const Reference& ref = Reference::fp32();
    const Trace& trace = tracedPrefill().trace;

    for (const char* name: {"cond.dataset_name", "cond.instrument_group"}) {
        const Tolerance tol = ref.tolerance(name);
        const Comparison comp = compare(trace.read(name), ref.tensor(name));
        INFO(describe(name, comp, tol));
        recordParity(name, comp, tol);
        CHECK(comp.max_abs <= tol.atol);
    }
}
