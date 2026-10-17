// ggml-vulkan resolves Vulkan through vulkan-hpp's dynamic dispatcher except for
// the four functions below, which it calls directly. Defining them here,
// forwarded to libvulkan.so.1 opened at first use, keeps the loader out of the
// binary's link dependencies, so it still loads on a system without Vulkan.

#include "vulkan_loader_linux.hpp"

#include <vulkan/vulkan.h>

#include <dlfcn.h>

namespace
{

struct Loader {
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2 = nullptr;
    PFN_vkCmdCopyBuffer cmd_copy_buffer = nullptr;

    bool available = false;
};

template <typename Function>
Function resolve(void* inHandle, const char* inName)
{
    return reinterpret_cast<Function>(dlsym(inHandle, inName));
}

const Loader& loader()
{
    static const Loader instance = [] {
        Loader result;
        void* const handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);

        if (handle == nullptr) {
            return result;
        }

        result.get_instance_proc_addr = resolve<PFN_vkGetInstanceProcAddr>(handle, "vkGetInstanceProcAddr");
        result.get_device_proc_addr = resolve<PFN_vkGetDeviceProcAddr>(handle, "vkGetDeviceProcAddr");
        result.get_physical_device_features2 =
            resolve<PFN_vkGetPhysicalDeviceFeatures2>(handle, "vkGetPhysicalDeviceFeatures2");
        result.cmd_copy_buffer = resolve<PFN_vkCmdCopyBuffer>(handle, "vkCmdCopyBuffer");

        result.available = result.get_instance_proc_addr != nullptr && result.get_device_proc_addr != nullptr
                           && result.get_physical_device_features2 != nullptr && result.cmd_copy_buffer != nullptr;
        return result;
    }();

    return instance;
}

} // namespace

namespace msl
{

bool vulkanLoaderAvailable()
{
    return loader().available;
}

} // namespace msl

// Only reached once vulkanLoaderAvailable() returned true.
extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance inInstance, const char* inName)
{
    return loader().get_instance_proc_addr(inInstance, inName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice inDevice, const char* inName)
{
    return loader().get_device_proc_addr(inDevice, inName);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice inPhysicalDevice,
                                                        VkPhysicalDeviceFeatures2* outFeatures)
{
    loader().get_physical_device_features2(inPhysicalDevice, outFeatures);
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer inCommandBuffer,
                                           VkBuffer inSrcBuffer,
                                           VkBuffer inDstBuffer,
                                           uint32_t inRegionCount,
                                           const VkBufferCopy* inRegions)
{
    loader().cmd_copy_buffer(inCommandBuffer, inSrcBuffer, inDstBuffer, inRegionCount, inRegions);
}

} // extern "C"
