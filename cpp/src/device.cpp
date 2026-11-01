#include "muscriptor/device.hpp"
#include "muscriptor/error.hpp"

#include "cpu_dispatch.hpp"
#include "device_backend.hpp"
#include "log.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#if defined(MUSCRIPTOR_HAS_METAL)
#include <ggml-metal.h>
#endif

#if defined(MUSCRIPTOR_HAS_VULKAN)
#include <ggml-vulkan.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#endif

namespace msl
{

namespace
{

#if defined(MUSCRIPTOR_HAS_VULKAN)
    /**
     * @return True when a Vulkan call is safe to make. On Windows
     *         `vulkan-1.dll` is delay-loaded and a missing loader would raise
     *         a structured exception on the first call.
     */
    bool vulkanLoaderPresent()
    {
#if defined(_WIN32)
        return LoadLibraryW(L"vulkan-1.dll") != nullptr;
#else
        return true;
#endif
    }
#endif

    /** `devices[i]` runs on `handles[i]`; the CPU's handle is null. */
    struct Registry {
        std::vector<Device> devices;
        std::vector<ggml_backend_dev_t> handles;

        void add(ggml_backend_dev_t inHandle, const char* inBackend)
        {
            ggml_backend_dev_props props {};
            ggml_backend_dev_get_props(inHandle, &props);

            // ggml-metal still creates a device when the system has no Metal
            // GPU, and leaves its description empty.
            if (props.description == nullptr || props.description[0] == '\0') {
                return;
            }

            devices.push_back({.name = props.description,
                               .backend = inBackend,
                               .integrated = props.type == GGML_BACKEND_DEVICE_TYPE_IGPU,
                               .memory_total = props.memory_total});
            handles.push_back(inHandle);
        }
    };

    /**
     * Each backend's own registry only: ggml's global one would load every
     * backend, and on Windows that faults without `vulkan-1.dll`.
     */
    Registry enumerate()
    {
        // Before the first ggml call of the process, so enumeration never
        // reaches ggml's own stderr handler.
        installLogHook();

        Registry registry;

#if defined(MUSCRIPTOR_HAS_METAL)
        // Only the physical GPU: GGML_METAL_DEVICES can add virtual devices on
        // top of it.
        if (ggml_backend_reg_t reg = ggml_backend_metal_reg(); reg != nullptr && ggml_backend_reg_dev_count(reg) > 0) {
            registry.add(ggml_backend_reg_dev_get(reg, 0), "Metal");
        }
#endif

#if defined(MUSCRIPTOR_HAS_VULKAN)
        if (vulkanLoaderPresent()) {
            // ggml_backend_vk_reg catches instance-creation failures; the device
            // queries after it are not guarded.
            try {
                if (ggml_backend_reg_t reg = ggml_backend_vk_reg(); reg != nullptr) {
                    const std::size_t count = ggml_backend_reg_dev_count(reg);

                    for (std::size_t i = 0; i < count; ++i) {
                        registry.add(ggml_backend_reg_dev_get(reg, i), "Vulkan");
                    }
                }
            } catch (...) {
            }
        }
#endif

        registry.devices.push_back({.name = "CPU", .backend = "CPU"});
        registry.handles.push_back(nullptr);
        return registry;
    }

    const Registry& registry()
    {
        static const Registry instance = enumerate();
        return instance;
    }

    /**
     * ggml's Vulkan device setup is lazily initialised without a lock, both by
     * a backend's creation and by a device query, and two models may load at
     * once.
     */
    std::mutex& deviceMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    /**
     * Devices whose creation threw. ggml-vulkan caches a device before
     * creating it, so using one that failed would reach a half-built device.
     * Guarded by deviceMutex.
     */
    std::set<ggml_backend_dev_t>& failedHandles()
    {
        static std::set<ggml_backend_dev_t> failed;
        return failed;
    }

    /** Thrown by firstUnsupportedNode when the query fails to create the device. */
    struct DeviceCreationFailed {
        std::string reason;
    };

    /** @return What a caught exception says, for a log line. */
    std::string reasonOf(const std::exception_ptr& inException)
    {
        try {
            std::rethrow_exception(inException);
        } catch (const std::exception& e) {
            return e.what();
        } catch (...) {
            return "unknown exception";
        }
    }

    /** The GPUs in the order Auto tries them: discrete ones, then integrated ones, each in list order. */
    std::vector<std::size_t> autoOrder(std::span<const Device> inDevices)
    {
        std::vector<std::size_t> order;

        for (const bool integrated: {false, true}) {
            for (std::size_t i = 0; i < inDevices.size(); ++i) {
                if (inDevices[i].backend != "CPU" && inDevices[i].integrated == integrated) {
                    order.push_back(i);
                }
            }
        }

        return order;
    }

    bool knownUnsupported(const std::optional<DeviceSupport>& inSupport)
    {
        return inSupport.has_value() && !inSupport->supported();
    }

    ggml_backend_dev_t handleOf(std::size_t inIndex)
    {
        const Registry& reg = registry();
        return reg.handles[inIndex] != nullptr ? reg.handles[inIndex]
                                               : ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    }

    /**
     * @return Whether device `inIndex` can run the model, or nullopt when it
     *         could not be created. Call with deviceMutex held.
     */
    std::optional<DeviceSupport> checkDevice(std::size_t inIndex, const SupportCheck& inCheck)
    {
        const std::string& name = registry().devices[inIndex].name;
        ggml_backend_dev_t handle = handleOf(inIndex);
        std::set<ggml_backend_dev_t>& failed = failedHandles();

        if (failed.contains(handle)) {
            return std::nullopt;
        }

        try {
            DeviceSupport support {.unsupported_op = inCheck(handle)};

            if (support.supported()) {
                logMessage(LogLevel::Debug, name + " can run the model");
            } else {
                logMessage(LogLevel::Info, name + " cannot run " + support.unsupported_op);
            }

            return support;
        } catch (const DeviceCreationFailed& e) {
            logMessage(LogLevel::Warn, "failed to create " + name + " while checking its ops: " + e.reason);
            failed.insert(handle);
            return std::nullopt;
        }
    }

    /**
     * @return The backend for device `inIndex`, or null if it cannot be
     *         initialised. Call with deviceMutex held.
     */
    ggml_backend_t initDevice(std::size_t inIndex)
    {
        const Registry& reg = registry();
        ggml_backend_dev_t handle = reg.handles[inIndex];

        if (handle == nullptr) {
            return ggml_backend_cpu_init();
        }

        const std::string& name = reg.devices[inIndex].name;
        std::set<ggml_backend_dev_t>& failed = failedHandles();

        if (failed.contains(handle)) {
            logMessage(LogLevel::Info, name + " failed to be created earlier in this process");
            return nullptr;
        }

        // Vulkan's device init has no failure path: a device that refuses
        // creation throws.
        try {
            if (ggml_backend_t backend = ggml_backend_dev_init(handle, nullptr); backend != nullptr) {
                return backend;
            }

            logMessage(LogLevel::Warn, "failed to initialise " + name);
        } catch (...) {
            logMessage(LogLevel::Warn, "failed to initialise " + name + ": " + reasonOf(std::current_exception()));
        }

        failed.insert(handle);
        return nullptr;
    }

} // namespace

std::vector<Device> availableDevices()
{
    return registry().devices;
}

std::size_t autoDevice(std::span<const Device> inDevices)
{
    const std::vector<std::size_t> order = autoOrder(inDevices);

    // The CPU is always last.
    return order.empty() ? inDevices.size() - 1 : order.front();
}

std::size_t autoDevice(std::span<const Device> inDevices, std::span<const std::optional<DeviceSupport>> inSupport)
{
    for (const std::size_t i: autoOrder(inDevices)) {
        if (i >= inSupport.size() || !knownUnsupported(inSupport[i])) {
            return i;
        }
    }

    return inDevices.size() - 1;
}

std::string_view cpuVariant()
{
#if defined(MUSCRIPTOR_CPU_DISPATCH)
    return cpuDispatch().name;
#else
    return "default";
#endif
}

void requireCpuBackend()
{
#if defined(MUSCRIPTOR_CPU_DISPATCH)
    if (const CpuDispatch& cpu = cpuDispatch(); cpu.name.empty()) {
        throw Exception(Error::UnsupportedCpu, cpu.reason);
    }
#endif
}

ggml_tensor* firstUnsupportedNode(ggml_backend_dev_t inDevice, ggml_cgraph* inGraph)
{
    // A Vulkan query creates the device, which throws if the device refuses.
    try {
        for (int i = 0; i < ggml_graph_n_nodes(inGraph); ++i) {
            if (ggml_tensor* node = ggml_graph_node(inGraph, i); !ggml_backend_dev_supports_op(inDevice, node)) {
                return node;
            }
        }
    } catch (...) {
        throw DeviceCreationFailed {reasonOf(std::current_exception())};
    }

    return nullptr;
}

std::vector<std::optional<DeviceSupport>> checkDeviceSupport(std::optional<std::size_t> inDevice,
                                                             const SupportCheck& inCheck)
{
    const Registry& reg = registry();
    std::vector<std::optional<DeviceSupport>> support(reg.devices.size());
    const std::lock_guard<std::mutex> lock(deviceMutex());

    if (inDevice.has_value()) {
        if (*inDevice >= reg.devices.size()) {
            throw Exception(Error::DeviceUnavailable, "no device at index " + std::to_string(*inDevice));
        }

        support[*inDevice] = checkDevice(*inDevice, inCheck);
        return support;
    }

    for (const std::size_t i: autoOrder(reg.devices)) {
        support[i] = checkDevice(i, inCheck);

        if (!knownUnsupported(support[i])) {
            return support;
        }
    }

    support.back() = checkDevice(reg.devices.size() - 1, inCheck);
    return support;
}

InitialisedBackend initBackend(std::optional<std::size_t> inDevice, const SupportCheck& inCheck)
{
    const Registry& reg = registry();
    const std::size_t cpu = reg.devices.size() - 1;

    const auto cannot_run = [&](std::size_t inIndex, const std::string& inOp) {
        return Exception(Error::DeviceUnavailable, reg.devices[inIndex].name + " cannot run " + inOp);
    };

    const std::lock_guard<std::mutex> lock(deviceMutex());

    if (inDevice.has_value()) {
        const std::size_t index = *inDevice;

        if (index >= reg.devices.size()) {
            throw Exception(Error::DeviceUnavailable, "no device at index " + std::to_string(index));
        }

        if (const auto support = checkDevice(index, inCheck); knownUnsupported(support)) {
            throw cannot_run(index, support->unsupported_op);
        }

        if (ggml_backend_t backend = initDevice(index); backend != nullptr) {
            return {backend, reg.devices[index]};
        }

        throw Exception(Error::DeviceUnavailable, "failed to initialise " + reg.devices[index].name);
    }

    // The first GPU that is not known to be unsupported is Auto's only GPU
    // attempt, as autoDevice(devices, support) names it.
    const std::vector<std::size_t> order = autoOrder(reg.devices);

    for (const std::size_t i: order) {
        if (knownUnsupported(checkDevice(i, inCheck))) {
            continue;
        }

        if (ggml_backend_t backend = initDevice(i); backend != nullptr) {
            return {backend, reg.devices[i]};
        }

        break;
    }

    if (!order.empty()) {
        logMessage(LogLevel::Info, "Auto falls back to the CPU");
    }

    if (const auto support = checkDevice(cpu, inCheck); knownUnsupported(support)) {
        throw cannot_run(cpu, support->unsupported_op);
    }

    if (ggml_backend_t backend = initDevice(cpu); backend != nullptr) {
        return {backend, reg.devices[cpu]};
    }

    throw Exception(Error::OutOfMemory, "failed to initialise the CPU backend");
}

} // namespace msl
