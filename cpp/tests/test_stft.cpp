// The STFT, from the waveform. The mel filterbank tests start from the dumped
// `in.spectrum` instead.
//
// Parity cases compare against `torch.stft` on the fixture chunks. Analytic
// cases pin the unnormalised scale, pffft's DC/Nyquist packing, and the window
// the checkpoint ships.

#include "muscriptor/stft.hpp"
#include "reference.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

using namespace msl;
using namespace msl::test;

namespace
{

// A rectangular-windowed transform, for the cases with a closed-form answer.
// The checkpoint's Hann window spreads a pure tone over three bins, which
// would turn every analytic expectation into an approximation.
Stft rectangularStft(int inNFft, int inHop)
{
    return Stft(inNFft, inHop, std::vector<float>(static_cast<std::size_t>(inNFft), 1.0f));
}

// A frame far enough from both ends that no reflected sample reaches it, so
// its contents are a plain slice of the input and the analytic cases can
// ignore padding entirely.
constexpr int INTERIOR_FRAME = 100;

} // namespace

TEST_CASE("stft magnitudes match torch.stft on every chunk", "[stft]")
{
    const Reference& ref = Reference::fp32();
    const Stft& stft = shared_model().stft();

    for (int chunk = 0; chunk < ref.num_chunks(); ++chunk) {
        const std::string wav_name = msl::format("in.chunk{}.wav", chunk);
        const std::string spec_name = msl::format("in.chunk{}.spectrum", chunk);

        const std::vector<float>& wav = ref.tensor(wav_name);
        const std::vector<float>& want = ref.tensor(spec_name);

        const std::vector<float> got = stft.magnitudes(wav);

        INFO("chunk " << chunk);
        REQUIRE(got.size() == want.size());

        const Tolerance tol = ref.tolerance(spec_name);
        const Comparison comp = compare(got, want);
        INFO(describe(spec_name, comp, tol));
        recordParity(spec_name, comp, tol);
        CHECK_FALSE(comp.nonfinite_mismatch);
        CHECK(comp.max_abs <= tol.atol);
        CHECK(comp.cosine >= tol.cosine_min);
    }
}

TEST_CASE("stft emits one more frame than the audio covers", "[stft]")
{
    const Reference& ref = Reference::fp32();
    const Stft& stft = shared_model().stft();

    // Centre padding is what produces the extra frame, and that extra frame is
    // exactly the one `encodeConditioning` masks away. If this ever comes out
    // one lower, the mask and the spectrum have silently gone out of step.
    CHECK(stft.nFrames(ref.segment_samples()) == ref.mel_frames());
    CHECK(stft.nFrames(ref.segment_samples()) == ref.segment_samples() / stft.hopLength() + 1);
    CHECK(stft.nFreq() == stft.nFft() / 2 + 1);
}

TEST_CASE("stft is unnormalised and unpacks DC and Nyquist", "[stft]")
{
    const Stft stft = rectangularStft(2048, 160);
    const int n_fft = stft.nFft();
    const int n_bin = stft.nFreq();
    const auto offset = static_cast<std::size_t>(INTERIOR_FRAME) * n_bin;
    const int n = 80000;

    // torch.stft(normalized=False) applies no 1/sqrt(N), so a DC signal puts
    // the full frame sum in bin 0 and nothing anywhere else.
    SECTION("DC lands in bin 0 with no scaling")
    {
        const std::vector<float> mag = stft.magnitudes(std::vector<float>(n, 1.0f));
        CHECK_THAT(mag[offset + 0], Catch::Matchers::WithinRel(static_cast<float>(n_fft), 1e-4f));
        CHECK(mag[offset + 1] < 1e-2f);
        CHECK(mag[offset + static_cast<std::size_t>(n_bin - 1)] < 1e-2f);
    }

    // The case pffft's packing gets wrong if bins[1] is mistaken for Im F(0):
    // an alternating signal is pure Nyquist, so all the energy belongs in the
    // last bin and none in the first.
    SECTION("an alternating signal is pure Nyquist")
    {
        std::vector<float> x(n);

        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 1.0f : -1.0f;
        }

        const std::vector<float> mag = stft.magnitudes(x);
        CHECK_THAT(mag[offset + static_cast<std::size_t>(n_bin - 1)],
                   Catch::Matchers::WithinRel(static_cast<float>(n_fft), 1e-4f));
        CHECK(mag[offset + 0] < 1e-2f);
    }

    // And a tone on an exact bin centre, which would still look right if the
    // bin indexing were off by the half-complex packing but the scale were not.
    SECTION("a bin-centred tone lands in that bin")
    {
        constexpr int bin = 64;
        std::vector<float> x(n);

        for (int i = 0; i < n; ++i) {
            const double phase = 2.0 * std::numbers::pi * bin * i / n_fft;
            x[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(phase));
        }

        const std::vector<float> mag = stft.magnitudes(x);
        CHECK_THAT(mag[offset + bin], Catch::Matchers::WithinRel(static_cast<float>(n_fft) / 2.0f, 1e-3f));
        CHECK(mag[offset + bin - 2] < 1.0f);
        CHECK(mag[offset + bin + 2] < 1.0f);
    }
}

TEST_CASE("the checkpoint's window is an fp16-rounded periodic Hann", "[stft]")
{
    const Stft& stft = shared_model().stft();
    const int n_fft = stft.nFft();

    // Periodic Hann (denominator n_fft, as torch.hann_window defaults to), stored
    // as its fp16 rounding: up to 2.4e-4 from the exact window. The reference
    // reads the same buffer.
    const std::span<const float> win = stft.window();
    REQUIRE(win.size() == static_cast<std::size_t>(n_fft));

    for (int i = 0; i < n_fft; ++i) {
        const double exact = 0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * i / static_cast<double>(n_fft)));
        const float want = ggml_fp16_to_fp32(ggml_fp32_to_fp16(static_cast<float>(exact)));
        REQUIRE_THAT(win[static_cast<std::size_t>(i)], Catch::Matchers::WithinAbs(want, 1e-7));
    }
}

TEST_CASE("encodeAudio is exactly the two stages composed", "[stft]")
{
    const Reference& ref = Reference::fp32();
    const std::vector<float>& wav = ref.tensor("in.chunk0.wav");
    const Model& model = shared_model();

    // Pure plumbing, so bitwise equality is the right bar -- anything looser
    // would be hiding a difference rather than measuring one.
    const std::vector<float> composed = model.encodeConditioning(
        model.stft().magnitudes(wav), model.stft().nFrames(static_cast<int>(wav.size())), static_cast<int>(wav.size()));
    const std::vector<float> direct = model.encodeAudio(wav);

    REQUIRE(direct.size() == composed.size());
    const auto mismatch = std::mismatch(direct.begin(), direct.end(), composed.begin());

    if (mismatch.first != direct.end()) {
        const auto index = static_cast<std::size_t>(mismatch.first - direct.begin());
        FAIL("element " << index << ": " << *mismatch.first << " != " << *mismatch.second);
    }
}

TEST_CASE("audio in, reference tokens out", "[stft][generate]")
{
    // The per-stage tolerances assume the dumped spectrum; an fp32 STFT moves
    // log-mel by more than `cond.logmel` allows, in PyTorch as here
    // (docs/TESTING.md). So from the waveform, tokens are compared instead.
    // Chunk 0 here; the Transcriber test covers every chunk.
    const Reference& ref = Reference::fp32();
    Model& model = shared_model();

    const std::vector<float> conditioning = model.encodeAudio(ref.tensor("in.chunk0.wav"));
    const std::vector<std::int32_t>& want = ref.tokens(0);
    const std::vector<std::int32_t> got =
        model.generate(conditioning, ref.mel_frames(), static_cast<int>(want.size()) + 8, ref.eos_id());

    REQUIRE(got.size() == want.size());

    for (std::size_t i = 0; i < want.size(); ++i) {
        INFO("token " << i);
        REQUIRE(got[i] == want[i]);
    }
}
