#include "cpu_dispatch.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#endif

// What each ggml-cpu variant defines, under the names the build renames them to
// (cmake/ggml_cpu_variants.cmake).
#define MUSCRIPTOR_DECLARE_CPU_VARIANT(tag)                                                                            \
    extern "C" ggml_backend_reg_t ggml_backend_cpu_reg__##tag(void);                                                   \
    extern "C" ggml_backend_t ggml_backend_cpu_init__##tag(void);                                                      \
    extern "C" bool ggml_backend_is_cpu__##tag(ggml_backend_t);                                                        \
    extern "C" void ggml_backend_cpu_set_n_threads__##tag(ggml_backend_t, int);                                        \
    extern "C" void ggml_backend_cpu_set_abort_callback__##tag(ggml_backend_t, ggml_abort_callback, void*);

MUSCRIPTOR_DECLARE_CPU_VARIANT(sse42)
MUSCRIPTOR_DECLARE_CPU_VARIANT(ivybridge)
MUSCRIPTOR_DECLARE_CPU_VARIANT(haswell)

namespace msl
{

namespace
{

    struct CpuFeatures {
        bool sse42 = false;
        bool popcnt = false;
        // False, with everything below that needs it, unless the OS also saves
        // the ymm registers.
        bool avx = false;
        bool f16c = false;
        bool fma = false;
        bool avx2 = false;
        bool bmi1 = false;
        bool bmi2 = false;
        bool lzcnt = false;
        bool movbe = false;
    };

    struct Variant {
        std::string_view tag;
        // The instruction sets its compile flags allow, for
        // `Error::UnsupportedCpu`'s message.
        std::string_view needs;
        bool (*runs)(const CpuFeatures&);

        ggml_backend_reg_t (*reg)();
        ggml_backend_t (*init)();
        bool (*is_cpu)(ggml_backend_t);
        void (*set_n_threads)(ggml_backend_t, int);
        void (*set_abort_callback)(ggml_backend_t, ggml_abort_callback, void*);
    };

#define MUSCRIPTOR_CPU_VARIANT_ENTRY_POINTS(tag)                                                                       \
    &ggml_backend_cpu_reg__##tag, &ggml_backend_cpu_init__##tag, &ggml_backend_is_cpu__##tag,                          \
        &ggml_backend_cpu_set_n_threads__##tag, &ggml_backend_cpu_set_abort_callback__##tag

    bool runsSse42(const CpuFeatures& inCpu)
    {
        return inCpu.sse42 && inCpu.popcnt;
    }

    bool runsIvyBridge(const CpuFeatures& inCpu)
    {
        return runsSse42(inCpu) && inCpu.avx && inCpu.f16c;
    }

    bool runsHaswell(const CpuFeatures& inCpu)
    {
        return runsIvyBridge(inCpu) && inCpu.avx2 && inCpu.fma && inCpu.bmi1 && inCpu.bmi2 && inCpu.lzcnt
               && inCpu.movbe;
    }

    // Best first.
    const std::array<Variant, 3> VARIANTS {{
        {"haswell",
         "AVX2, FMA, F16C, BMI1, BMI2, LZCNT and MOVBE",
         &runsHaswell,
         MUSCRIPTOR_CPU_VARIANT_ENTRY_POINTS(haswell)},
        {"ivybridge", "AVX and F16C", &runsIvyBridge, MUSCRIPTOR_CPU_VARIANT_ENTRY_POINTS(ivybridge)},
        {"sse42", "SSE4.2 and POPCNT", &runsSse42, MUSCRIPTOR_CPU_VARIANT_ENTRY_POINTS(sse42)},
    }};

    struct CpuidRegisters {
        std::uint32_t eax = 0;
        std::uint32_t ebx = 0;
        std::uint32_t ecx = 0;
        std::uint32_t edx = 0;
    };

    CpuidRegisters cpuid(std::uint32_t inLeaf)
    {
#if defined(_MSC_VER)
        std::array<int, 4> regs {};
        __cpuidex(regs.data(), static_cast<int>(inLeaf), 0);
        return {static_cast<std::uint32_t>(regs[0]),
                static_cast<std::uint32_t>(regs[1]),
                static_cast<std::uint32_t>(regs[2]),
                static_cast<std::uint32_t>(regs[3])};
#else
        CpuidRegisters regs;
        __asm__ __volatile__("cpuid"
                             : "=a"(regs.eax), "=b"(regs.ebx), "=c"(regs.ecx), "=d"(regs.edx)
                             : "a"(inLeaf), "c"(0));
        return regs;
#endif
    }

    /** XCR0: the register state the OS saves. Only valid when CPUID reports OSXSAVE. */
    std::uint64_t xcr0()
    {
#if defined(_MSC_VER)
        return _xgetbv(0);
#else
        std::uint32_t low = 0;
        std::uint32_t high = 0;
        __asm__ __volatile__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
    }

    bool bit(std::uint32_t inRegister, int inBit)
    {
        return ((inRegister >> inBit) & 1U) != 0;
    }

    CpuFeatures detect()
    {
        const std::uint32_t max_leaf = cpuid(0).eax;
        const std::uint32_t max_extended_leaf = cpuid(0x80000000U).eax;

        const CpuidRegisters leaf1 = max_leaf >= 1 ? cpuid(1) : CpuidRegisters {};
        const CpuidRegisters leaf7 = max_leaf >= 7 ? cpuid(7) : CpuidRegisters {};
        const CpuidRegisters leaf81 = max_extended_leaf >= 0x80000001U ? cpuid(0x80000001U) : CpuidRegisters {};

        // XMM and YMM state, bits 1 and 2.
        const bool os_saves_ymm = bit(leaf1.ecx, 27) && (xcr0() & 0x6U) == 0x6U;

        CpuFeatures cpu;
        cpu.sse42 = bit(leaf1.ecx, 20);
        cpu.popcnt = bit(leaf1.ecx, 23);
        cpu.avx = bit(leaf1.ecx, 28) && os_saves_ymm;
        cpu.f16c = bit(leaf1.ecx, 29) && cpu.avx;
        cpu.fma = bit(leaf1.ecx, 12) && cpu.avx;
        cpu.movbe = bit(leaf1.ecx, 22);
        cpu.avx2 = bit(leaf7.ebx, 5) && cpu.avx;
        cpu.bmi1 = bit(leaf7.ebx, 3);
        cpu.bmi2 = bit(leaf7.ebx, 8);
        cpu.lzcnt = bit(leaf81.ecx, 5);
        return cpu;
    }

    std::string variantList()
    {
        std::string list;

        for (const Variant& variant: VARIANTS) {
            list += (list.empty() ? "" : ", ") + std::string(variant.tag);
        }

        return list;
    }

    struct Selection {
        const Variant* variant = nullptr;
        CpuDispatch dispatch;
    };

    Selection chooseVariant()
    {
        const CpuFeatures cpu = detect();

        if (const char* forced = std::getenv("MUSCRIPTOR_CPU_VARIANT"); forced != nullptr && forced[0] != '\0') {
            for (const Variant& variant: VARIANTS) {
                if (variant.tag != forced) {
                    continue;
                }

                if (!variant.runs(cpu)) {
                    return {nullptr,
                            {"",
                             "MUSCRIPTOR_CPU_VARIANT=" + std::string(forced) + " needs " + std::string(variant.needs)
                                 + ", which this CPU lacks"}};
                }

                return {&variant, {variant.tag, ""}};
            }

            return {nullptr, {"", "MUSCRIPTOR_CPU_VARIANT=" + std::string(forced) + " is not one of " + variantList()}};
        }

        for (const Variant& variant: VARIANTS) {
            if (variant.runs(cpu)) {
                return {&variant, {variant.tag, ""}};
            }
        }

        return {nullptr,
                {"", "this CPU lacks " + std::string(VARIANTS.back().needs) + ", the minimum the CPU backend needs"}};
    }

    const Selection& selection()
    {
        static const Selection instance = chooseVariant();
        return instance;
    }

    /** Null when no variant can run: then no CPU backend exists either. */
    const Variant* variant()
    {
        return selection().variant;
    }

} // namespace

const CpuDispatch& cpuDispatch()
{
    return selection().dispatch;
}

} // namespace msl

// The ggml-cpu functions the rest of the build calls. Everything else in
// ggml-cpu is reached through the backend and registry these return.

ggml_backend_reg_t ggml_backend_cpu_reg(void)
{
    const msl::Variant* variant = msl::variant();
    return variant != nullptr ? variant->reg() : nullptr;
}

ggml_backend_t ggml_backend_cpu_init(void)
{
    const msl::Variant* variant = msl::variant();
    return variant != nullptr ? variant->init() : nullptr;
}

bool ggml_backend_is_cpu(ggml_backend_t inBackend)
{
    const msl::Variant* variant = msl::variant();
    return variant != nullptr && variant->is_cpu(inBackend);
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t inBackend, int inThreads)
{
    if (const msl::Variant* variant = msl::variant(); variant != nullptr) {
        variant->set_n_threads(inBackend, inThreads);
    }
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t inBackend, ggml_abort_callback inCallback, void* inCallbackData)
{
    if (const msl::Variant* variant = msl::variant(); variant != nullptr) {
        variant->set_abort_callback(inBackend, inCallback, inCallbackData);
    }
}
