#pragma once

namespace msl
{

/**
 * @return True when `libvulkan.so.1` opened and exports every Vulkan function
 *         ggml-vulkan calls directly. No Vulkan call may be made otherwise.
 */
bool vulkanLoaderAvailable();

} // namespace msl
