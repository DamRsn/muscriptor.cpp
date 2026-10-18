#pragma once

#include <string>
#include <string_view>

namespace msl
{

/** The ggml-cpu variant this process runs, or why it runs none. */
struct CpuDispatch {
    /** `"haswell"`, `"ivybridge"` or `"sse42"`; empty when no variant can run. */
    std::string_view tag;
    /** Why no variant can run, for `Error::UnsupportedCpu`. Empty otherwise. */
    std::string reason;
};

/**
 * Chooses on the first call, from CPUID and the OS's register support, or from
 * `MUSCRIPTOR_CPU_VARIANT`. Fixed for the life of the process. Thread-safe.
 * Only in builds with MUSCRIPTOR_CPU_VARIANTS.
 */
const CpuDispatch& cpuDispatch();

} // namespace msl
