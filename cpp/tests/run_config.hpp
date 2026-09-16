#pragma once

#include "format.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace msl::test
{

enum class Device { cpu, gpu };

/**
 * One configuration of the ladder: which converted checkpoint it loads and
 * which backend runs it. Set by main() from the command line, once per
 * session when `--size all` or `--device all` expands into several.
 */
struct RunConfig {
    std::string size = "medium";
    Device device = Device::cpu;
    std::string weight_dtype = "f16";

    bool operator==(const RunConfig&) const = default;

    /** @return "medium/cpu/f16", the form the summary table and headers use. */
    std::string label() const;

    /** @return `testdata/refs/<size>`, where the dump for this size lives. */
    std::filesystem::path refsDir() const;

    /** @return The converted weights this configuration loads. */
    std::filesystem::path weightsPath() const;

    /** @return The `uv run ...` line that produces this size's reference dump. */
    std::string refsHint() const;

    /** @return The `uv run ...` line that obtains these weights. */
    std::string weightsHint() const;
};

/** Where the fixtures live. Overridable with MUSCRIPTOR_TESTDATA. */
std::filesystem::path testdataRoot();

/**
 * Format a double the way a printf conversion would, for diagnostic strings.
 *
 * Not `std::format`, which libc++ gates on a macOS 13.3 deployment target
 * (see format.hpp), and not `msl::format`, which has no format specs.
 *
 * @param inSpec A printf conversion including its `%`, e.g. "%.3f" or "%.6e".
 */
std::string formatDouble(const char* inSpec, double inValue);

const RunConfig& currentConfig();
void setCurrentConfig(RunConfig inConfig);

class PerConfigBase
{
public:
    virtual ~PerConfigBase() = default;
    virtual void release() = 0;

protected:
    PerConfigBase();
};

/**
 * Releases every PerConfig value. main() calls it before returning: ggml's
 * Metal device registry is a static constructed during the first load, so it
 * is destroyed before the function-local statics holding the models, and
 * asserts if any of their buffers are still alive.
 */
void releasePerConfigValues();

/**
 * A value built once per configuration: rebuilt when the configuration
 * changes, and the previous one released first, so this instance never holds
 * two checkpoints at once.
 */
template <typename T>
class PerConfig : public PerConfigBase
{
public:
    T& get(const std::function<std::unique_ptr<T>()>& inMake)
    {
        if (!mValue || mConfig != currentConfig()) {
            mValue.reset();
            mValue = inMake();
            mConfig = currentConfig();
        }

        return *mValue;
    }

    void release() override { mValue.reset(); }

private:
    std::unique_ptr<T> mValue;
    RunConfig mConfig;
};

} // namespace msl::test
