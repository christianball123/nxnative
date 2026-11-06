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
    X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdClearDepthStencilImage) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkCreateBufferView) X(vkDestroyBufferView) X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreatePipelineCache) X(vkGetPipelineCacheData) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDispatch) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetStencilReference) \
    X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) X(vkCmdSetBlendConstants) X(vkCmdSetDepthBias) X(vkFlushMappedMemoryRanges) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)
#define VK_DECL(f) extern PFN_##f f##_;
VKI_FUNCS(VK_DECL) VKD_FUNCS(VK_DECL)
#undef VK_DECL

typedef struct {
    VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q; uint32_t qfam;
    VkPhysicalDeviceMemoryProperties mem;
    pthread_mutex_t qmu;   // the queue is used from the submit thread and the present thread: Vulkan wants one at a time
    int ok;
    int push_desc, f_dynrender;   // VK_KHR_push_descriptor is on; dynamic rendering (Vulkan 1.3) is on
    int depth_clip_control;       // VK_EXT_depth_clip_control: clip depth in [-1, 1] (NVN's default GL depth mode)
    int f_logic_op, f_dual_src, f_depth_clamp, f_wide_lines;
    VkPhysicalDeviceLimits limits;
} vk_core;
extern vk_core VK;

int vkw_wanted(void);           // DM_WINDOW is set (to anything but 0)
int vkw_active(void);           // the window and device are up
int vkw_init(void);             // make the window and device now (any thread; once); 1 when up
// Show present number n: the game's image (VK_IMAGE_LAYOUT_GENERAL, its extent w x h, crop x, y, w, h or NULL) scaled into the window, or stage A's colour when img is null
void vkw_present(uint64_t n, VkImage img, uint32_t w, uint32_t h, const int crop[4]);
// The game's image is sRGB-encoded (RGBA8_SRGB, Mario Tennis Fever's swapchain): the window's swapchain takes the same encoding, so the blit that copies it decodes and encodes
// again and the colours arrive unchanged (an sRGB image blitted into a UNORM swapchain is decoded once and shown too dark and saturated)
void vkw_image_srgb(int srgb);
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want);   // a memory type index of those bits with the properties, or ~0u

// Stage C, shaders (gfx_vk_shader.c): the game's shaders translated to SPIR-V, from the pack dm/linux/tools/spv_pack.py writes (DM_SHADERS = its path)
typedef struct { uint8_t binding, vktype, kind, image; uint32_t reg; uint16_t count, aux; } vks_bind;   // spv_pack.py's binding record
enum { VKS_SUPPORT, VKS_CBUF, VKS_SBUF, VKS_TEX, VKS_IMG, VKS_UNKNOWN };
typedef struct { uint64_t key; uint8_t model, nb, nvin, fout; uint16_t local[3], c1_len; const vks_bind *b; const uint8_t *vin, *c1; const uint32_t *spv; uint32_t nspv; } vks_shader;
enum { VKS_VS, VKS_TCS, VKS_TES, VKS_GS, VKS_FS, VKS_NGFX, VKS_CS = 5 };   // the SPIR-V execution models
// Each stage's descriptor set: vertex 0, fragment 1, geometry 2, tessellation control 3, evaluation 4; compute 0. A vertex + fragment program uses two sets (a GPU must allow four)
static inline int vks_set_of(int model) { static const int8_t s[6] = {0, 3, 4, 2, 1, 0}; return model >= 0 && model < 6 ? s[model] : 0; }
typedef struct { uint64_t serial; int compute, ok, nset, has_gs, gs_class; const void *prog; const vks_shader *st[6]; VkShaderModule mod[6]; VkDescriptorSetLayout set[5]; VkPipelineLayout layout; } vks_prog;   // set[k]: descriptor set k (vks_set_of)
int vks_init(void);                                  // the pack is open (1), or there is none (0: draws are skipped, as before stage C)
const vks_shader *vks_find(uint64_t key);
struct prog_t_;
const vks_prog *vks_prog_get(const void *prog);     // the program's shaders, modules and layout (cached by its serial); NULL until its shaders are known
void vks_stats(uint64_t *progs, uint64_t *missing, uint64_t *failed);
const vks_prog *vks_prog_topology(const vks_prog *base, int cls);   // a program with a geometry shader, made for an input topology class (others: base itself)
_Static_assert(sizeof(vks_bind) == 12, "vks_bind is spv_pack.py record");
