#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace msl
{

/** A device the model can run on. */
struct Device {
    /** Human-readable, e.g. `"NVIDIA GeForce RTX 4070"`, `"Apple M1 Pro"`, `"CPU"`. Not unique. */
    std::string name;

    /** `"Metal"`, `"Vulkan"` or `"CPU"`. Stable: a host may branch on it. */
    std::string backend;

    /** ggml reports it as an integrated GPU. Only Vulkan makes the distinction. */
    bool integrated = false;

    /** Bytes of memory the device can use, 0 when unknown (always for the CPU). */
    std::size_t memory_total = 0;
};

/**
 * Whether a device can run a checkpoint's model, as its backend reports for
 * each op of the model's graphs. See `Transcriber::checkDevices`.
 */
struct DeviceSupport {
    /** The first op the device cannot run, e.g. `"MUL_MAT (f16, f32)"`; empty when it runs them all. */
    std::string unsupported_op;

    bool supported() const { return unsupported_op.empty(); }
};

/**
 * The devices this build can run on, GPUs first in the backend's own order,
 * the CPU always last, so the list is never empty.
 *
 * The list is built on the first call and fixed for the life of the process:
 * ggml enumerates its GPUs once. The first call can take a noticeable time
 * (Vulkan instance creation, Metal shader compilation); keep it off a UI thread.
 * Thread-safe.
 */
std::vector<Device> availableDevices();

/**
 * The device Auto tries when every GPU can run the checkpoint: the first
 * discrete GPU, otherwise the first integrated GPU, otherwise the CPU. Once
 * the checkpoint is known, prefer the overload that takes its support.
 *
 * @param inDevices A list from `availableDevices`.
 * @return An index into `inDevices`.
 */
std::size_t autoDevice(std::span<const Device> inDevices);

/**
 * The device `LoadOptions::device` left empty tries for a checkpoint: as
 * `autoDevice(inDevices)`, among the GPUs `inSupport` marks supported.
 *
 * @param inDevices A list from `availableDevices`.
 * @param inSupport From `Transcriber::checkDevices`, one entry per device.
 * @return An index into `inDevices`.
 */
std::size_t autoDevice(std::span<const Device> inDevices, std::span<const DeviceSupport> inSupport);

/**
 * The instruction-set variant of the CPU backend this process runs. On an
 * x86-64 build with MUSCRIPTOR_CPU_VARIANTS, the best of `"x86-64-v3"`,
 * `"x86-64-v2+f16c"` and `"x86-64-v2"` the CPU supports, or the one the
 * `MUSCRIPTOR_CPU_VARIANT` environment variable names. `"default"` on every
 * other build.
 *
 * Empty when the CPU supports none of them, or not the one named: every load
 * then fails with `Error::UnsupportedCpu`, whatever the device. Chosen on the
 * first call and fixed for the life of the process. Thread-safe.
 */
std::string_view cpuVariant();

} // namespace msl
