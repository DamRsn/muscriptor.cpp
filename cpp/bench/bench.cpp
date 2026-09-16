// Decode-loop benchmark. Times the three phases of a chunk separately --
// conditioning, prefill, and greedy decode -- and prints the token ids it
// produced, so an optimisation can be checked for "same tokens, less time"
// in one run instead of through the full ladder.

#include "muscriptor/model.hpp"
#include "muscriptor/transcriber.hpp"

#include "format.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

double millisSince(Clock::time_point inStart)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - inStart).count();
}

/**
 * Minimal reader for the one fixture this benchmark consumes: 16 kHz mono
 * float32 WAV. Not a general decoder -- anything else is an error rather than
 * a conversion, because a silently resampled fixture would make every number
 * below meaningless.
 *
 * @param inPath File to read.
 * @return The samples.
 */
std::vector<float> readFloatWav(const std::filesystem::path& inPath)
{
    std::ifstream file(inPath, std::ios::binary);

    if (!file) {
        throw std::runtime_error(msl::format("cannot open {}", inPath.string()));
    }

    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0
        || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("not a RIFF/WAVE file");
    }

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t rate = 0;
    std::uint16_t bits = 0;
    std::vector<float> samples;

    for (std::size_t pos = 12; pos + 8 <= bytes.size();) {
        std::uint32_t size = 0;
        std::memcpy(&size, bytes.data() + pos + 4, 4);
        const char* id = bytes.data() + pos;
        const std::size_t body = pos + 8;

        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16 && body + 16 <= bytes.size()) {
            std::memcpy(&format, bytes.data() + body + 0, 2);
            std::memcpy(&channels, bytes.data() + body + 2, 2);
            std::memcpy(&rate, bytes.data() + body + 4, 4);
            std::memcpy(&bits, bytes.data() + body + 14, 2);
        }

        else if (std::memcmp(id, "data", 4) == 0) {
            const std::size_t count = std::min<std::size_t>(size, bytes.size() - body) / sizeof(float);
            samples.resize(count);
            std::memcpy(samples.data(), bytes.data() + body, count * sizeof(float));
        }

        pos = body + size + (size & 1u);
    }

    if (format != 3 || channels != 1 || bits != 32 || rate != 16000) {
        throw std::runtime_error(msl::format(
            "expected 16 kHz mono float32, got format={} ch={} rate={} bits={}", format, channels, rate, bits));
    }

    return samples;
}

struct Args {
    std::string size = MUSCRIPTOR_TEST_SIZE;
    std::string weight_dtype = MUSCRIPTOR_TEST_WEIGHT_DTYPE;
    // Set explicitly with --weights; otherwise derived from size and dtype.
    std::filesystem::path weights;
    std::filesystem::path audio;
    int steps = 64;
    int threads = 0;
    int chunk = 0;
    int repeats = 1;
    bool use_gpu = true;
    bool transcribe = false;
};

/** @return Percentile `inQ` of an already-sorted `inSorted`. */
double percentile(const std::vector<double>& inSorted, double inQ)
{
    if (inSorted.empty()) {
        return 0.0;
    }

    const std::size_t index =
        std::min(inSorted.size() - 1, static_cast<std::size_t>(inQ * static_cast<double>(inSorted.size())));
    return inSorted[index];
}

constexpr const char* USAGE = "usage: %s [--size small|medium|large] [--device cpu|gpu] [--weight-dtype f16|f32]\n"
                              "       [--weights f.gguf] [--audio f.wav] [--steps N] [--threads N]\n"
                              "       [--chunk K] [--repeats N] [--transcribe]\n";

/**
 * Parse the command line.
 *
 * Throws rather than exiting, so a bad value lands in the same handler as a
 * bad fixture and prints the same way -- an uncaught throw out of here would
 * abort the process instead.
 *
 * @param inArgc Argument count, as passed to main.
 * @param inArgv Argument vector, as passed to main.
 * @return The parsed arguments, or nullopt when an unrecognised flag made this
 *         print the usage text instead.
 */
std::optional<Args> parseArgs(int inArgc, char** inArgv)
{
    const char* testdata_env = std::getenv("MUSCRIPTOR_TESTDATA");
    const std::filesystem::path testdata =
        testdata_env != nullptr ? std::filesystem::path(testdata_env) : std::filesystem::path(MUSCRIPTOR_TESTDATA_DIR);

    Args args;
    args.use_gpu = std::string_view(MUSCRIPTOR_TEST_DEVICE) == "gpu";
    args.audio = testdata / "audio" / "fixture_3chunks_16k.wav";

    for (int i = 1; i < inArgc; ++i) {
        const std::string flag = inArgv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= inArgc) {
                throw std::runtime_error(msl::format("{} needs a value", flag));
            }

            return inArgv[++i];
        };

        // Negative values are rejected after the loop; a negative --chunk would
        // wrap the unsigned offset arithmetic.
        const auto count = [&]() {
            const std::string value = next();

            try {
                return std::stoi(value);
            }

            catch (const std::exception&) {
                throw std::runtime_error(msl::format("{} needs a number, got '{}'", flag, value));
            }
        };

        if (flag == "--size") {
            args.size = next();
        }

        else if (flag == "--weight-dtype") {
            args.weight_dtype = next();
        }

        else if (flag == "--device") {
            const std::string value = next();

            if (value != "cpu" && value != "gpu") {
                throw std::runtime_error(msl::format("--device must be cpu or gpu, got '{}'", value));
            }

            args.use_gpu = value == "gpu";
        }

        else if (flag == "--weights") {
            args.weights = next();
        }

        else if (flag == "--audio") {
            args.audio = next();
        }

        else if (flag == "--steps") {
            args.steps = count();
        }

        else if (flag == "--threads") {
            args.threads = count();
        }

        else if (flag == "--chunk") {
            args.chunk = count();
        }

        else if (flag == "--repeats") {
            args.repeats = count();
        }

        else if (flag == "--transcribe") {
            args.transcribe = true;
        }

        else {
            std::fprintf(stderr, USAGE, inArgv[0]);
            return std::nullopt;
        }
    }

    if (args.steps < 0 || args.threads < 0 || args.chunk < 0) {
        throw std::runtime_error("--steps, --threads and --chunk cannot be negative");
    }

    if (args.repeats < 1) {
        throw std::runtime_error("--repeats must be at least 1");
    }

    if (args.weights.empty()) {
        args.weights = testdata / "weights" / msl::format("muscriptor-{}-{}.gguf", args.size, args.weight_dtype);
    }

    return args;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::optional<Args> parsed = parseArgs(argc, argv);

        if (!parsed.has_value()) {
            return 2;
        }

        const Args& args = *parsed;
        const std::vector<float> signal = readFloatWav(args.audio);

        // Whole-signal mode: the number a caller actually feels, rather than
        // the per-phase breakdown the decode work is tuned against.
        if (args.transcribe) {
            // This mode runs the whole signal through the public API, so the
            // knobs that carve up a single chunk have nothing to act on. Say so
            // rather than silently reporting a number the flags did not shape.
            if (args.steps != Args {}.steps || args.chunk != 0 || args.repeats != 1) {
                std::fprintf(stderr, "note: --steps, --chunk and --repeats do not apply to --transcribe\n");
            }

            msl::LoadOptions load_options;
            load_options.use_gpu = args.use_gpu;

            Clock::time_point began = Clock::now();
            std::expected<msl::Transcriber, msl::Error> transcriber =
                msl::Transcriber::load(args.weights, load_options);

            if (!transcriber.has_value()) {
                throw std::runtime_error(msl::format("load failed: {}", msl::describe(transcriber.error())));
            }

            const double loaded_ms = millisSince(began);

            msl::TranscribeOptions transcribe_options;
            transcribe_options.n_threads = args.threads;

            began = Clock::now();
            const std::expected<std::vector<msl::Note>, msl::Error> notes =
                transcriber->transcribe(signal, transcribe_options);
            const double elapsed_ms = millisSince(began);

            if (!notes.has_value()) {
                throw std::runtime_error(msl::format("transcribe failed: {}", msl::describe(notes.error())));
            }

            const double audio_s = static_cast<double>(signal.size()) / msl::Transcriber::SAMPLE_RATE;

            std::printf("weights   %s\n", args.weights.filename().string().c_str());
            std::printf("backend   %8s\n", transcriber->backendName());
            std::printf("load      %8.1f ms\n", loaded_ms);
            std::printf("audio     %8.2f s   (%d chunks)\n", audio_s, msl::Transcriber::chunkCount(signal.size()));
            std::printf("transcribe %7.2f s\n", elapsed_ms / 1000.0);
            std::printf("speed     %8.2fx real time\n", audio_s / (elapsed_ms / 1000.0));
            std::printf("notes     %8zu\n", notes->size());
            return 0;
        }

        const std::size_t offset = static_cast<std::size_t>(args.chunk) * msl::Transcriber::SEGMENT_SAMPLES;

        if (offset + msl::Transcriber::SEGMENT_SAMPLES > signal.size()) {
            throw std::runtime_error(msl::format("chunk {} is past the end of {} samples", args.chunk, signal.size()));
        }

        const std::span<const float> chunk(signal.data() + offset, msl::Transcriber::SEGMENT_SAMPLES);

        Clock::time_point start = Clock::now();
        msl::Model::Options options;
        options.n_threads = args.threads;
        options.use_gpu = args.use_gpu;
        msl::Model model = msl::Model::load(args.weights, options);
        const double load_ms = millisSince(start);

        std::printf("weights   %s\n", args.weights.filename().string().c_str());
        std::printf("backend   %8s\n", model.backendName());
        if (args.threads > 0) {
            std::printf("threads   %8d\n", args.threads);
        }

        else {
            std::printf("threads      auto\n");
        }

        std::printf("load      %8.1f ms\n", load_ms);

        std::vector<double> encode_ms;
        std::vector<double> prefill_ms;
        std::vector<double> decode_ms;
        std::vector<std::int32_t> tokens;

        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            model.reset();

            start = Clock::now();
            const std::vector<float> cond = model.encodeAudio(chunk);
            encode_ms.push_back(millisSince(start));

            const int n_frames = static_cast<int>(cond.size() / static_cast<std::size_t>(model.hparams().dim));
            const std::int32_t initial = model.hparams().initial_token_id;

            start = Clock::now();
            std::vector<float> logits = model.prefill(cond, n_frames, std::span(&initial, 1));
            prefill_ms.push_back(millisSince(start));

            tokens.clear();

            for (int step = 0; step < args.steps; ++step) {
                const auto best = std::max_element(logits.begin(), logits.end());
                const std::int32_t next = static_cast<std::int32_t>(std::distance(logits.begin(), best));
                tokens.push_back(next);

                start = Clock::now();
                logits = model.decode(next);
                decode_ms.push_back(millisSince(start));
            }
        }

        std::vector<double> sorted = decode_ms;
        std::sort(sorted.begin(), sorted.end());
        const double total = std::accumulate(decode_ms.begin(), decode_ms.end(), 0.0);
        const double mean = decode_ms.empty() ? 0.0 : total / static_cast<double>(decode_ms.size());

        const auto best_of = [](const std::vector<double>& v) {
            return v.empty() ? 0.0 : *std::ranges::min_element(v);
        };

        std::printf("encode    %8.1f ms   (best of %zu)\n", best_of(encode_ms), encode_ms.size());
        std::printf("prefill   %8.1f ms   (best of %zu)\n", best_of(prefill_ms), prefill_ms.size());
        std::printf("decode    %8.2f ms/step   min %.2f  p50 %.2f  p90 %.2f  max %.2f   over %zu steps\n",
                    mean,
                    best_of(decode_ms),
                    percentile(sorted, 0.50),
                    percentile(sorted, 0.90),
                    sorted.empty() ? 0.0 : sorted.back(),
                    decode_ms.size());
        std::printf("decode    %8.1f tok/s\n", mean > 0.0 ? 1000.0 / mean : 0.0);
        std::printf("n_past    %8d\n", model.nPast());

        // The correctness half of the benchmark: an optimisation is only an
        // optimisation if this line does not move.
        std::printf("tokens   ");

        for (std::size_t i = 0; i < tokens.size() && i < 24; ++i) {
            std::printf(" %d", tokens[i]);
        }

        std::printf("%s\n", tokens.size() > 24 ? " ..." : "");

        std::uint64_t digest = 1469598103934665603ull;

        for (const std::int32_t token: tokens) {
            digest = (digest ^ static_cast<std::uint64_t>(token)) * 1099511628211ull;
        }

        std::printf("digest    %016llx  (%zu tokens)\n", static_cast<unsigned long long>(digest), tokens.size());
    }

    catch (const std::exception& e) {
        std::fprintf(stderr, "bench failed: %s\n", e.what());
        return 1;
    }

    return 0;
}
