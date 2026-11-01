#pragma once

#include "muscriptor/device.hpp"

#include <ggml-backend.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace msl
{

struct InitialisedBackend {
    ggml_backend_t backend = nullptr;
    Device device;
};

/** Throws `Error::UnsupportedCpu` when no CPU backend can run on this CPU. */
void requireCpuBackend();

/**
 * @return The first node of `inGraph` that `inDevice` cannot run, or null.
 *         Call only from a `SupportCheck`: on Vulkan the query creates the
 *         device, and the caller of the check handles that failing.
 */
ggml_tensor* firstUnsupportedNode(ggml_backend_dev_t inDevice, ggml_cgraph* inGraph);

/** Given a device, returns the first op of the model it cannot run, or an empty string. */
using SupportCheck = std::function<std::string(ggml_backend_dev_t inDevice)>;

/**
 * Checks every one of `availableDevices()`.
 *
 * @return One entry per device. A device that cannot even be created is
 *         reported as supported: `initBackend` is what fails it.
 */
std::vector<DeviceSupport> checkDeviceSupport(const SupportCheck& inCheck);

/**
 * Initialises the backend for one of `availableDevices()`, as described at
 * `LoadOptions::device`, checking only the devices it tries. Naming a device
 * that cannot run the model throws `Error::DeviceUnavailable`; Auto passes
 * over it.
 *
 * @param inDevice An index into `availableDevices()`, or empty for Auto.
 * @return A backend the caller owns, and the device it runs on.
 */
InitialisedBackend initBackend(std::optional<std::size_t> inDevice, const SupportCheck& inCheck);

} // namespace msl
