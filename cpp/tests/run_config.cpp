#include "run_config.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace msl::test
{

namespace
{
    RunConfig& mutableConfig()
    {
        static RunConfig config;
        return config;
    }

    std::vector<PerConfigBase*>& registry()
    {
        static std::vector<PerConfigBase*> instances;
        return instances;
    }
} // namespace

PerConfigBase::PerConfigBase()
{
    registry().push_back(this);
}

void releasePerConfigValues()
{
    for (PerConfigBase* instance: registry()) {
        instance->release();
    }
}

std::string RunConfig::label() const
{
    return msl::format("{}/{}/{}", size, device == Device::gpu ? "gpu" : "cpu", weight_dtype);
}

std::filesystem::path RunConfig::refsDir() const
{
    return testdataRoot() / "refs" / size;
}

std::filesystem::path RunConfig::weightsPath() const
{
    return testdataRoot() / "weights" / msl::format("muscriptor-{}-{}.gguf", size, weight_dtype);
}

std::string RunConfig::refsHint() const
{
    return msl::format("uv run msl-dump-refs --size {}", size);
}

std::string RunConfig::weightsHint() const
{
    // Only F16 is published; F32 is a local diagnostic and has to be converted.
    if (weight_dtype == "f16") {
        return msl::format("uv run msl-fetch --size {}", size);
    }
    return msl::format("uv run msl-convert --size {} --weight-dtype {}", size, weight_dtype);
}

std::filesystem::path testdataRoot()
{
    if (const char* override_path = std::getenv("MUSCRIPTOR_TESTDATA")) {
        return std::filesystem::path(override_path);
    }

    return std::filesystem::path(MUSCRIPTOR_TESTDATA_DIR);
}

std::string formatDouble(const char* inSpec, double inValue)
{
    // Wide enough for any conversion used here; %.12f of a large double is the
    // longest, and snprintf truncates rather than overruns in any case.
    std::array<char, 64> buffer {};
    const int written = std::snprintf(buffer.data(), buffer.size(), inSpec, inValue);

    if (written < 0) {
        return {};
    }

    return std::string(buffer.data(), static_cast<std::size_t>(std::min<int>(written, buffer.size() - 1)));
}

const RunConfig& currentConfig()
{
    return mutableConfig();
}

void setCurrentConfig(RunConfig inConfig)
{
    mutableConfig() = std::move(inConfig);
}

} // namespace msl::test
