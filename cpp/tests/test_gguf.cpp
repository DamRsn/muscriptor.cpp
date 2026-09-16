// The fixtures load and describe the same model the reference ran.
//
// Everything downstream assumes the converted weights and the dumped reference
// agree on the architecture. Checking that here means a stale GGUF fails with
// "24 layers vs 14" rather than with an inscrutable numerical mismatch fifty
// tensors later.

#include "reference.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace msl;
using namespace msl::test;

TEST_CASE("weights and reference describe the same architecture", "[gguf]")
{
    const Reference& ref = Reference::fp32();
    const Hparams& hp = shared_model().hparams();

    CHECK(hp.n_layer == ref.n_layer());
    CHECK(hp.initial_token_id == ref.initial_token_id());
    CHECK(hp.head_dim * hp.n_head == hp.dim);

    // The conditioning prefix is the mel frames plus the two class embeddings.
    CHECK(ref.prepend_length() == ref.mel_frames() + 2);

    // Front-end constants carried through the conversion.
    CHECK(hp.sample_rate == 16000);
    CHECK(hp.n_fft == 2048);
    CHECK(hp.hop_length == 160);
    CHECK(hp.n_mels == 512);
    CHECK(hp.n_freq() == 1025);
}

TEST_CASE("reference tensors have the shapes the port expects", "[gguf]")
{
    const Reference& ref = Reference::fp32();
    const Hparams& hp = shared_model().hparams();

    CHECK(ref.shape("in.spectrum") == std::vector<std::int64_t> {hp.n_freq(), ref.mel_frames()});
    CHECK(ref.shape("cond.logmel") == std::vector<std::int64_t> {hp.n_mels, ref.mel_frames()});
    CHECK(ref.shape("cond.embed") == std::vector<std::int64_t> {hp.dim, ref.mel_frames()});
    CHECK(ref.shape("pre.layer_in") == std::vector<std::int64_t> {hp.dim, ref.prepend_length() + 1});
    CHECK(ref.shape("post.logits") == std::vector<std::int64_t> {hp.vocab_size});

    for (int il = 0; il < ref.n_layer(); ++il) {
        INFO("layer " << il);
        CHECK(ref.has(msl::format("blk.{}.out", il)));
    }
}

TEST_CASE("the sinusoidal position table matches the reference", "[gguf][positions]")
{
    // Not a weight but a formula, and one with two easy mistakes baked in (the
    // half-1 denominator, the cosine-first ordering), so it gets checked on its
    // own before it can silently poison every layer.
    const Reference& ref = Reference::fp32();
    const Model& model = shared_model();
    const Hparams& hp = model.hparams();

    const std::vector<float>& want = ref.tensor("pre.pos_emb");
    const std::span<const float> table = model.positionEmbeddings();
    const std::vector<float> got(table.begin(), table.begin() + static_cast<std::ptrdiff_t>(want.size()));
    REQUIRE(static_cast<int>(want.size()) == (ref.prepend_length() + 1) * hp.dim);

    // Held to an fp64 evaluation rather than to the manifest's 1e-5 floor: the
    // phase reaches ~500 rad, where one fp32 ulp already exceeds it.
    const int rows = ref.prepend_length() + 1;
    const int half = hp.dim / 2;
    std::vector<float> truth(want.size());

    for (int pos = 0; pos < rows; ++pos) {
        for (int i = 0; i < half; ++i) {
            const double phase = pos / std::pow(static_cast<double>(hp.max_period), i / (half - 1.0));
            truth[static_cast<std::size_t>(pos * hp.dim + i)] = static_cast<float>(std::cos(phase));
            truth[static_cast<std::size_t>(pos * hp.dim + half + i)] = static_cast<float>(std::sin(phase));
        }
    }

    const float max_phase = static_cast<float>(rows - 1);
    const double ulp = std::nextafter(max_phase, 2.0f * max_phase) - max_phase;
    const Tolerance versus_truth {2.0 * ulp, ref.tolerance("pre.pos_emb").cosine_min};
    const Tolerance versus_reference {4.0 * ulp, ref.tolerance("pre.pos_emb").cosine_min};

    const Comparison to_truth = compare(got, truth);
    INFO(describe("pre.pos_emb (fp64 truth)", to_truth, versus_truth));
    CHECK_FALSE(to_truth.nonfinite_mismatch);
    CHECK(to_truth.max_abs <= versus_truth.atol);
    CHECK(to_truth.cosine >= versus_truth.cosine_min);

    const Comparison c = compare(got, want);
    INFO(describe("pre.pos_emb", c, versus_reference));
    recordParity("pre.pos_emb", c, versus_reference);
    CHECK_FALSE(c.nonfinite_mismatch);
    CHECK(c.max_abs <= versus_reference.atol);
    CHECK(c.cosine >= versus_reference.cosine_min);
}
