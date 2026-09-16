// Entry point: Catch2's own, plus the three options that pick which checkpoint
// and backend the ladder runs against. `all` for --size or --device expands into
// one Catch2 session per combination, run back to back in this process, with a
// summary at the end.

#include "reference.hpp"
#include "run_config.hpp"

#include <catch2/catch_session.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace msl::test;

namespace
{

constexpr std::array<const char*, 3> SIZES {"small", "medium", "large"};
constexpr std::array<const char*, 2> DEVICES {"cpu", "gpu"};
constexpr std::array<const char*, 2> WEIGHT_DTYPES {"f16", "f32"};

/**
 * @return The values `inChoice` selects out of `inAll`, or empty when it names
 *         none of them.
 */
template <std::size_t N>
std::vector<std::string> expand(const std::string& inChoice, const std::array<const char*, N>& inAll)
{
    if (inChoice == "all") {
        return {inAll.begin(), inAll.end()};
    }

    if (std::find(inAll.begin(), inAll.end(), inChoice) != inAll.end()) {
        return {inChoice};
    }

    return {};
}

/**
 * @return Why `inConfig` cannot run, or empty when it can. `inExpanded` says
 *         whether it came out of an `all`: a missing fixture then skips the
 *         combination, whereas one asked for by name runs and fails with the
 *         generating command.
 */
std::string skipReason(const RunConfig& inConfig, bool inExpanded)
{
#if defined(MUSCRIPTOR_HAS_METAL) || defined(MUSCRIPTOR_HAS_VULKAN)
    // Both GPU backends round some F32 operands to half, so an all-F32 build on
    // the GPU measures the backend, not the port (docs/PERFORMANCE.md).
    if (inConfig.device == Device::gpu && inConfig.weight_dtype == "f32") {
        return "f32 weights are a CPU diagnostic; the GPU backends round F32 operands to half";
    }
#endif

    if (inExpanded) {
        // large is verified on the GPU only: a 1.4B-parameter ladder on the CPU
        // runs for the better part of an hour. Ask for it by name to run it.
        if (inConfig.size == "large" && inConfig.device == Device::cpu) {
            return "large is verified on the GPU only; run --size large --device cpu by name to override";
        }

        if (!std::filesystem::exists(inConfig.refsDir() / "manifest.json")) {
            return "no reference dump; generate it with:\n    " + inConfig.refsHint();
        }

        if (!std::filesystem::exists(inConfig.weightsPath())) {
            return "no weights; get them with:\n    " + inConfig.weightsHint();
        }
    }

    return {};
}

struct Outcome {
    RunConfig config;
    int failed = 0;
    std::string skipped;
};

// Catch2 returns a failure count, so this only has to be non-zero; the stderr
// line is what tells the two apart.
constexpr int NOTHING_RAN = 3;

} // namespace

int main(int argc, char* argv[])
{
    Catch::Session session;

    std::string size = MUSCRIPTOR_TEST_SIZE;
    // Empty until --device is given, so the default can depend on the weights.
    std::string device;
    std::string weight_dtype = MUSCRIPTOR_TEST_WEIGHT_DTYPE;
    std::string parity_report;
    bool full = false;

    using namespace Catch::Clara;
    session.cli(session.cli() | Opt(size, "all|small|medium|large")["--size"]("checkpoint to run the ladder against")
                | Opt(device, "all|cpu|gpu")["--device"]("backend to run it on; defaults to " MUSCRIPTOR_TEST_DEVICE
                                                         ", or cpu for f32 weights")
                | Opt(weight_dtype, "f16|f32")["--weight-dtype"]("converted weights to load; f32 is a CPU diagnostic")
                | Opt(parity_report, "file")["--parity-report"](
                    "write every tensor comparison as JSON lines; render with uv run msl-parity-report")
                | Opt(full)["--full"]("run the [slow] tests too; implied by any explicit test spec"));

    if (const int rc = session.applyCommandLine(argc, argv); rc != 0) {
        return rc;
    }

    // The GPU backends round F32 operands to half, so f32 weights only mean
    // something on the CPU.
    if (device.empty()) {
        device = weight_dtype == "f32" ? "cpu" : MUSCRIPTOR_TEST_DEVICE;
    }

    const std::vector<std::string> sizes = expand(size, SIZES);
    const std::vector<std::string> devices = expand(device, DEVICES);
    const std::vector<std::string> dtypes =
        std::find(WEIGHT_DTYPES.begin(), WEIGHT_DTYPES.end(), weight_dtype) != WEIGHT_DTYPES.end()
            ? std::vector<std::string> {weight_dtype}
            : std::vector<std::string> {};

    if (sizes.empty() || devices.empty() || dtypes.empty()) {
        std::fprintf(stderr,
                     "unrecognised --size '%s', --device '%s' or --weight-dtype '%s'\n",
                     size.c_str(),
                     device.c_str(),
                     weight_dtype.c_str());
        return 2;
    }

    const bool expanded = sizes.size() * devices.size() * dtypes.size() > 1;

    // Listing tests, tags or reporters needs no configuration and must not
    // repeat per combination.
    const Catch::ConfigData& data = session.configData();

    if (data.listTests || data.listTags || data.listReporters || data.listListeners || data.showHelp) {
        return session.run();
    }

    if (!parity_report.empty()) {
        setParityReportPath(parity_report);
    }

    // The [slow] tests re-transcribe the whole fixture several times over.
    // A bare run leaves them out so the ladder stays short enough to run on
    // every change; --full, or naming any test or tag, puts them back.
    if (!full && session.configData().testsOrTags.empty()) {
        session.configData().testsOrTags.push_back("~[slow]");
    }

    std::vector<Outcome> outcomes;

    for (const std::string& s: sizes) {
        for (const std::string& d: devices) {
            for (const std::string& w: dtypes) {
                Outcome outcome;
                outcome.config = RunConfig {s, d == "gpu" ? Device::gpu : Device::cpu, w};
                outcome.skipped = skipReason(outcome.config, expanded);

                if (expanded) {
                    std::printf("\n=== %s ===\n", outcome.config.label().c_str());
                }

                if (!outcome.skipped.empty()) {
                    std::printf("skipped %s: %s\n", outcome.config.label().c_str(), outcome.skipped.c_str());
                }

                else {
                    setCurrentConfig(outcome.config);
                    outcome.failed = session.run();
                }

                outcomes.push_back(std::move(outcome));
            }
        }
    }

    releasePerConfigValues();

    if (expanded) {
        std::printf("\n%-20s %s\n", "configuration", "result");

        for (const Outcome& outcome: outcomes) {
            const std::string result = !outcome.skipped.empty() ? "skipped"
                                       : outcome.failed == 0    ? "passed"
                                                                : std::to_string(outcome.failed) + " failed";
            std::printf("%-20s %s\n", outcome.config.label().c_str(), result.c_str());
        }
    }

    // Nothing ran, so there is no result to report: 0 here would read as a pass.
    if (std::all_of(
            outcomes.begin(), outcomes.end(), [](const Outcome& inOutcome) { return !inOutcome.skipped.empty(); })) {
        std::fprintf(stderr, "\nno configuration ran\n");
        return NOTHING_RAN;
    }

    int worst = 0;

    for (const Outcome& outcome: outcomes) {
        worst = std::max(worst, outcome.failed);
    }

    return worst;
}
