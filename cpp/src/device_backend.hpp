#pragma once

#include "muscriptor/device.hpp"

#include <ggml-backend.h>

#include <cstddef>
#include <optional>

namespace msl
{

struct InitialisedBackend {
    ggml_backend_t backend = nullptr;
    Device device;
};

/**
 * Initialises the backend for one of `availableDevices()`.
 *
 * An explicit device is a requirement: if it cannot be initialised this throws
 * `Error::DeviceUnavailable` rather than running somewhere else. Auto tries
 * `autoDevice`'s pick and falls back to the CPU.
 *
 * @param inDevice Index into `availableDevices()`, or empty for Auto.
 * @return A backend the caller owns, and the device it runs on.
 */
InitialisedBackend initBackend(std::optional<std::size_t> inDevice);

} // namespace msl
