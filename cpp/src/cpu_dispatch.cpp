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

// Each variant's ggml_backend_cpu_reg, under the name the build renames it to
// (cmake/ggml_cpu_variants.cmake).
extern "C" ggml_backend_reg_t ggml_backend_cpu_reg__v2(void);
extern "C" ggml_backend_reg_t ggml_backend_cpu_reg__f16c(void);
extern "C" ggml_backend_reg_t ggml_backend_cpu_reg__v3(void);

namespace msl
{

namespace
{

    struct CpuFeatures {
        bool sse3 = false;
        bool ssse3 = false;
        bool sse41 = false;
        bool sse42 = false;
        bool popcnt = false;
        bool cmpxchg16b = false;
        bool lahf = false;
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
        // What `cpuVariant()` returns.
        std::string_view name;
        // The instruction sets its compile flags allow, for
        // `Error::UnsupportedCpu`'s message.
        std::string_view needs;
        bool (*runs)(const CpuFeatures&);
        ggml_backend_reg_t (*reg)();
    };

    // The x86-64 psABI levels, whose lists -march=x86-64-v2 and -v3 compile for.
    bool runsV2(const CpuFeatures& inCpu)
    {
        return inCpu.sse3 && inCpu.ssse3 && inCpu.sse41 && inCpu.sse42 && inCpu.popcnt && inCpu.cmpxchg16b
               && inCpu.lahf;
    }

    bool runsF16c(const CpuFeatures& inCpu)
    {
        return runsV2(inCpu) && inCpu.avx && inCpu.f16c;
    }

    bool runsV3(const CpuFeatures& inCpu)
    {
        return runsF16c(inCpu) && inCpu.avx2 && inCpu.fma && inCpu.bmi1 && inCpu.bmi2 && inCpu.lzcnt && inCpu.movbe;
    }

    // Best first.
    const std::array<Variant, 3> VARIANTS {{
        {"x86-64-v3", "AVX2, FMA, F16C, BMI1, BMI2, LZCNT and MOVBE", &runsV3, &ggml_backend_cpu_reg__v3},
        {"x86-64-v2+f16c", "AVX and F16C", &runsF16c, &ggml_backend_cpu_reg__f16c},
        {"x86-64-v2",
         "SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B and LAHF/SAHF",
         &runsV2,
         &ggml_backend_cpu_reg__v2},
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
        cpu.sse3 = bit(leaf1.ecx, 0);
        cpu.ssse3 = bit(leaf1.ecx, 9);
        cpu.cmpxchg16b = bit(leaf1.ecx, 13);
        cpu.sse41 = bit(leaf1.ecx, 19);
        cpu.sse42 = bit(leaf1.ecx, 20);
        cpu.popcnt = bit(leaf1.ecx, 23);
        cpu.lahf = bit(leaf81.ecx, 0);
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
            list += (list.empty() ? "" : ", ") + std::string(variant.name);
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
                if (variant.name != forced) {
                    continue;
                }

                if (!variant.runs(cpu)) {
                    return {nullptr,
                            {"",
                             "MUSCRIPTOR_CPU_VARIANT=" + std::string(forced) + " needs " + std::string(variant.needs)
                                 + ", which this CPU lacks"}};
                }

                return {&variant, {variant.name, ""}};
            }

            return {nullptr, {"", "MUSCRIPTOR_CPU_VARIANT=" + std::string(forced) + " is not one of " + variantList()}};
        }

        for (const Variant& variant: VARIANTS) {
            if (variant.runs(cpu)) {
                return {&variant, {variant.name, ""}};
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

// The ggml-cpu functions the rest of the build calls. Only the registry comes
// from the variant; the others go through it.

ggml_backend_reg_t ggml_backend_cpu_reg(void)
{
    const msl::Variant* variant = msl::variant();
    return variant != nullptr ? variant->reg() : nullptr;
}

ggml_backend_t ggml_backend_cpu_init(void)
{
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    return reg != nullptr ? ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr) : nullptr;
}

bool ggml_backend_is_cpu(ggml_backend_t inBackend)
{
    ggml_backend_reg_t reg = ggml_backend_cpu_reg();
    return reg != nullptr && inBackend != nullptr
           && ggml_backend_get_device(inBackend) == ggml_backend_reg_dev_get(reg, 0);
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t inBackend, int inThreads)
{
    if (ggml_backend_reg_t reg = ggml_backend_cpu_reg(); reg != nullptr) {
        auto set = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        set(inBackend, inThreads);
    }
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t inBackend, ggml_abort_callback inCallback, void* inCallbackData)
{
    if (ggml_backend_reg_t reg = ggml_backend_cpu_reg(); reg != nullptr) {
        auto set = reinterpret_cast<ggml_backend_set_abort_callback_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback"));
        set(inBackend, inCallback, inCallbackData);
    }
}
