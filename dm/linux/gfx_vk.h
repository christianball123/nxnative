// The Vulkan graphics layer's shared state (dm/linux/GRAPHICS.md): the device the window module (gfx_vk_window.c) makes, its functions, and the entry points for the rest of the host.
#pragma once
#include <stdint.h>
#include <pthread.h>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif
#include <vulkan/vulkan_core.h>

#define VKI_FUNCS(X) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceImageFormatProperties)
#define VKD_FUNCS(X) X(vkGetDeviceQueue) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdClearColorImage) \
    X(vkQueueSubmit) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkDeviceWaitIdle) X(vkQueueWaitIdle) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindImageMemory) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdClearDepthStencilImage)
#define VK_DECL(f) extern PFN_##f f##_;
VKI_FUNCS(VK_DECL) VKD_FUNCS(VK_DECL)
#undef VK_DECL

typedef struct {
    VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mem;
    pthread_mutex_t qmu;   // the queue is used from the submit thread and the present thread: Vulkan wants one at a time
    int ok;
} vk_core;
extern vk_core VK;

int vkw_wanted(void);           // DM_WINDOW is set (to anything but 0)
int vkw_active(void);           // the window and device are up
int vkw_init(void);             // make the window and device now (any thread; once); 1 when up
// Show present number n: the game's image (VK_IMAGE_LAYOUT_GENERAL, its extent w x h, crop x, y, w, h or NULL) scaled into the window, or stage A's colour when img is null
void vkw_present(uint64_t n, VkImage img, uint32_t w, uint32_t h, const int crop[4]);
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want);   // a memory type index of those bits with the properties, or ~0u
