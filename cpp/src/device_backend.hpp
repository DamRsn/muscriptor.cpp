#pragma once

#include "muscriptor/device.hpp"

#include <ggml-backend.h>

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
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
 * Asks each of `availableDevices()` whether it can run a model.
 *
 * @param inFirstUnsupportedOp Given a device, returns the first op it cannot
 *        run, or an empty string.
 * @return One entry per device. A device that cannot even be created is
 *         reported as supported: `initBackend` is what fails it.
 */
std::vector<DeviceSupport>
    checkDeviceSupport(const std::function<std::string(ggml_backend_dev_t inDevice)>& inFirstUnsupportedOp);

/**
 * Initialises the backend for one of `availableDevices()`, as described at
 * `LoadOptions::device`. A device `inSupport` marks unsupported is never used:
 * Auto passes over it, and naming it throws `Error::DeviceUnavailable`.
 *
 * @param inDevice An index into `availableDevices()`, or empty for Auto.
 * @param inSupport One entry per device, from checking the checkpoint.
 * @return A backend the caller owns, and the device it runs on.
 */
InitialisedBackend initBackend(std::optional<std::size_t> inDevice, std::span<const DeviceSupport> inSupport);

} // namespace msl
