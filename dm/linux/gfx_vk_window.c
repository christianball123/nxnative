// Stage A of the Vulkan graphics layer (dm/linux/GRAPHICS.md): a window (SDL3) and a Vulkan swapchain that presents once per game present, at the game's pace.
// SDL3 and the Vulkan loader are opened at run time (dlopen), so the build needs neither's development files: ArmadaOS has libSDL3.so.0 and libvulkan.so.1 (sudo dnf install SDL3 vulkan-loader
// in the toolbox if not). Until stage B shows the game's own image, the window shows a colour that steps with every present, so its pace and the frame count are visible.
// The window is made on the thread of the first present (the game's present thread), and its events are read on that thread at every present.
#define VK_NO_PROTOTYPES 1
#include <vulkan/vulkan_core.h>   // the repository's headers (third_party/vulkan-headers): core with the KHR surface and swapchain
#include "dm.h"
#include "gfx_vk.h"
#include <dlfcn.h>
#include <math.h>
#include <stdatomic.h>

// ---- SDL3, the part used (declared here: no SDL headers needed)
typedef struct SDL_Window SDL_Window;
typedef union { uint32_t type; uint8_t pad[128]; } SDL_Event;
enum { SDL_INIT_VIDEO = 0x20, SDL_INIT_GAMEPAD = 0x2000, SDL_INIT_EVENTS = 0x4000 };
enum { SDL_WINDOW_FULLSCREEN = 0x1, SDL_WINDOW_RESIZABLE = 0x20, SDL_WINDOW_VULKAN = 0x10000000 };
enum { SDL_EVENT_QUIT = 0x100, SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED = 0x207, SDL_EVENT_KEY_DOWN = 0x300 };
static struct {
    void *lib;
    bool (*Init)(uint32_t);
    const char *(*GetError)(void);
    SDL_Window *(*CreateWindow)(const char *, int, int, uint64_t);
    bool (*SetWindowTitle)(SDL_Window *, const char *);
    bool (*GetWindowSizeInPixels)(SDL_Window *, int *, int *);
    bool (*PollEvent)(SDL_Event *);
    const char *const *(*Vulkan_GetInstanceExtensions)(uint32_t *);
    bool (*Vulkan_CreateSurface)(SDL_Window *, VkInstance, const VkAllocationCallbacks *, VkSurfaceKHR *);
    void *(*Vulkan_GetVkGetInstanceProcAddr)(void);
    bool (*Vulkan_LoadLibrary)(const char *);
} S;

static int sdl_load(void) {
    const char *names[] = {"libSDL3.so.0", "libSDL3.so", NULL};
    for (int i = 0; names[i] && !S.lib; i++) S.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!S.lib) { LOG("[RED] [vk] window: SDL3 is not installed (libSDL3.so.0: %s). In the toolbox: sudo dnf install -y SDL3", dlerror()); return 0; }
#define L(f) if (!(*(void **)&S.f = dlsym(S.lib, "SDL_" #f))) { LOG("[RED] [vk] window: SDL3 has no SDL_" #f); return 0; }
    L(Init) L(GetError) L(CreateWindow) L(SetWindowTitle) L(GetWindowSizeInPixels) L(PollEvent) L(Vulkan_GetInstanceExtensions) L(Vulkan_CreateSurface) L(Vulkan_GetVkGetInstanceProcAddr) L(Vulkan_LoadLibrary)
#undef L
    return 1;
}

// ---- Vulkan functions, loaded through SDL's vkGetInstanceProcAddr
#define VKI_FUNCS(X) X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceMemoryProperties)
#define VKD_FUNCS(X) X(vkGetDeviceQueue) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdClearColorImage) \
    X(vkQueueSubmit) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkDeviceWaitIdle)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
static PFN_vkCreateInstance vkCreateInstance_;
#define DECL(f) static PFN_##f f##_;
VKI_FUNCS(DECL) VKD_FUNCS(DECL)
#undef DECL

#define SWAP_MAX 8
static struct {
    SDL_Window *win;
    VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q; uint32_t qfam; VkSurfaceKHR surf;
    VkSwapchainKHR sc; VkFormat fmt; VkExtent2D ext; uint32_t nimg; VkImage img[SWAP_MAX];
    VkSemaphore acquired[SWAP_MAX], done[SWAP_MAX]; VkCommandPool pool; VkCommandBuffer cb[SWAP_MAX]; VkFence fence[SWAP_MAX]; uint32_t slot;
    int ok, failed, resized; uint64_t frames; double t0; char devname[256]; int ext_host_mem;
} W;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static int swapchain_make(void) {
    VkSurfaceCapabilitiesKHR cap; vkGetPhysicalDeviceSurfaceCapabilitiesKHR_(W.phys, W.surf, &cap);
    int pw = 0, ph = 0; S.GetWindowSizeInPixels(W.win, &pw, &ph);
    VkExtent2D e = cap.currentExtent.width != 0xffffffffu ? cap.currentExtent : (VkExtent2D){(uint32_t)pw, (uint32_t)ph};
    if (!e.width || !e.height) return 0;   // minimized
    uint32_t n = 0; vkGetPhysicalDeviceSurfaceFormatsKHR_(W.phys, W.surf, &n, NULL);
    VkSurfaceFormatKHR f[64]; if (n > 64) n = 64; vkGetPhysicalDeviceSurfaceFormatsKHR_(W.phys, W.surf, &n, f);
    VkSurfaceFormatKHR pick = f[0];
    for (uint32_t i = 0; i < n; i++) if (f[i].format == VK_FORMAT_B8G8R8A8_UNORM || f[i].format == VK_FORMAT_R8G8B8A8_UNORM) { pick = f[i]; break; }
    uint32_t count = cap.minImageCount + 1; if (cap.maxImageCount && count > cap.maxImageCount) count = cap.maxImageCount; if (count > SWAP_MAX) count = SWAP_MAX;
    VkSwapchainKHR old = W.sc;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, W.surf, count, pick.format, pick.colorSpace, e, 1,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
                                   cap.currentTransform, VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_TRUE, old};
    if (!(cap.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) ci.compositeAlpha = (VkCompositeAlphaFlagBitsKHR)(cap.supportedCompositeAlpha & -cap.supportedCompositeAlpha);
    VkResult r = vkCreateSwapchainKHR_(W.dev, &ci, NULL, &W.sc);
    if (old) { vkDeviceWaitIdle_(W.dev); vkDestroySwapchainKHR_(W.dev, old, NULL); }
    if (r) { LOG("[RED] [vk] vkCreateSwapchainKHR %dx%d failed (%d)", e.width, e.height, r); W.sc = VK_NULL_HANDLE; return 0; }
    W.fmt = pick.format; W.ext = e; W.nimg = SWAP_MAX; vkGetSwapchainImagesKHR_(W.dev, W.sc, &W.nimg, W.img);
    LOG("[vk] swapchain %ux%u, %u images, format %d, FIFO (vsync)", e.width, e.height, W.nimg, pick.format);
    return 1;
}

static int window_init(void) {
    if (!sdl_load()) return 0;
    if (!S.Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) { LOG("[RED] [vk] SDL_Init: %s", S.GetError()); return 0; }
    if (!S.Vulkan_LoadLibrary(NULL)) { LOG("[RED] [vk] no Vulkan loader (%s). In the toolbox: sudo dnf install -y vulkan-loader mesa-vulkan-drivers", S.GetError()); return 0; }
    const char *full = getenv("DM_FULLSCREEN");
    W.win = S.CreateWindow("nxnative", 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | (full && *full == '1' ? SDL_WINDOW_FULLSCREEN : 0));
    if (!W.win) { LOG("[RED] [vk] SDL_CreateWindow: %s", S.GetError()); return 0; }
    vkGetInstanceProcAddr_ = (PFN_vkGetInstanceProcAddr)S.Vulkan_GetVkGetInstanceProcAddr();
    vkCreateInstance_ = (PFN_vkCreateInstance)vkGetInstanceProcAddr_(NULL, "vkCreateInstance");
    uint32_t next = 0; const char *const *exts = S.Vulkan_GetInstanceExtensions(&next);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative", 1, "nxnative", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, next, exts};
    VkResult r = vkCreateInstance_(&ici, NULL, &W.inst);
    if (r) { LOG("[RED] [vk] vkCreateInstance failed (%d)", r); return 0; }
#define LI(f) f##_ = (PFN_##f)vkGetInstanceProcAddr_(W.inst, #f); if (!f##_) { LOG("[RED] [vk] no " #f); return 0; }
    VKI_FUNCS(LI)
#undef LI
    if (!S.Vulkan_CreateSurface(W.win, W.inst, NULL, &W.surf)) { LOG("[RED] [vk] SDL_Vulkan_CreateSurface: %s", S.GetError()); return 0; }
    uint32_t np = 8; VkPhysicalDevice pd[8]; vkEnumeratePhysicalDevices_(W.inst, &np, pd);
    for (uint32_t i = 0; i < np && !W.phys; i++) {
        uint32_t nq = 16; VkQueueFamilyProperties qp[16]; vkGetPhysicalDeviceQueueFamilyProperties_(pd[i], &nq, qp);
        for (uint32_t k = 0; k < nq; k++) {
            VkBool32 pres = 0; vkGetPhysicalDeviceSurfaceSupportKHR_(pd[i], k, W.surf, &pres);
            if ((qp[k].queueFlags & VK_QUEUE_GRAPHICS_BIT) && pres) { W.phys = pd[i]; W.qfam = k; break; }
        }
    }
    if (!W.phys) { LOG("[RED] [vk] no GPU can draw to this window"); return 0; }
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties_(W.phys, &pp);
    snprintf(W.devname, sizeof W.devname, "%s", pp.deviceName);
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties_(W.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep); vkEnumerateDeviceExtensionProperties_(W.phys, NULL, &ne, ep);
    for (uint32_t i = 0; i < ne; i++) if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) W.ext_host_mem = 1;
    free(ep);
    LOG("[vk] GPU %s, Vulkan %u.%u.%u, driver %#x; guest memory import (VK_EXT_external_memory_host): %s", pp.deviceName, VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion),
        VK_API_VERSION_PATCH(pp.apiVersion), pp.driverVersion, W.ext_host_mem ? "yes" : "no (stage B will mirror pools instead)");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, W.qfam, 1, &prio};
    const char *dext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 1, dext, NULL};
    if ((r = vkCreateDevice_(W.phys, &dci, NULL, &W.dev))) { LOG("[RED] [vk] vkCreateDevice failed (%d)", r); return 0; }
#define LD(f) f##_ = (PFN_##f)vkGetDeviceProcAddr_(W.dev, #f); if (!f##_) { LOG("[RED] [vk] no " #f); return 0; }
    VKD_FUNCS(LD)
#undef LD
    vkGetDeviceQueue_(W.dev, W.qfam, 0, &W.q);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, W.qfam};
    vkCreateCommandPool_(W.dev, &pci, NULL, &W.pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, W.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, SWAP_MAX};
    vkAllocateCommandBuffers_(W.dev, &cai, W.cb);
    for (int i = 0; i < SWAP_MAX; i++) {
        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, NULL, 0};
        vkCreateSemaphore_(W.dev, &si, NULL, &W.acquired[i]); vkCreateSemaphore_(W.dev, &si, NULL, &W.done[i]);
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT};
        vkCreateFence_(W.dev, &fi, NULL, &W.fence[i]);
    }
    if (!swapchain_make()) return 0;
    W.t0 = now_s();
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier_(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}

static void events(void) {
    SDL_Event e;
    while (S.PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) { LOG("[vk] window closed: quitting"); fflush(NULL); _exit(0); }
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) W.resized = 1;
    }
}

int vkw_active(void) { return W.ok; }
int vkw_wanted(void) { const char *w = getenv("DM_WINDOW"); return w && *w && *w != '0'; }

// One present: the next swapchain image, cleared to this frame's colour (stage A), shown with FIFO (the display's vsync paces the game as a Switch's display would)
void vkw_present(uint64_t n) {
    if (W.failed) return;
    if (!W.ok) { if (!window_init()) { W.failed = 1; LOG("[RED] [vk] no window: running headless"); return; } W.ok = 1; }
    events();
    if (W.resized || !W.sc) { W.resized = 0; if (!swapchain_make()) return; }
    uint32_t s = W.slot = (W.slot + 1) % W.nimg;
    vkWaitForFences_(W.dev, 1, &W.fence[s], VK_TRUE, UINT64_MAX);
    uint32_t idx;
    VkResult r = vkAcquireNextImageKHR_(W.dev, W.sc, UINT64_MAX, W.acquired[s], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.resized = 1; return; }
    if (r && r != VK_SUBOPTIMAL_KHR) { LOG("[RED] [vk] vkAcquireNextImageKHR failed (%d)", r); return; }
    vkResetFences_(W.dev, 1, &W.fence[s]);
    VkCommandBuffer cb = W.cb[s]; vkResetCommandBuffer_(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vkBeginCommandBuffer_(cb, &bi);
    barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    // Stage A's picture: a dark blue that brightens and dims once a second at 60 presents a second (so a steady pulse means a steady 60 fps)
    double ph = (double)(n % 60) / 60.0, k = 0.5 - 0.5 * cos(ph * 2 * M_PI);
    VkClearColorValue c = {{0.02f + 0.10f * (float)k, 0.06f + 0.20f * (float)k, 0.20f + 0.45f * (float)k, 1.0f}};
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage_(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rr);
    barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer_(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &W.acquired[s], &ws, 1, &cb, 1, &W.done[s]};
    if ((r = vkQueueSubmit_(W.q, 1, &si, W.fence[s]))) { LOG("[RED] [vk] vkQueueSubmit failed (%d)", r); return; }
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &W.done[s], 1, &W.sc, &idx, NULL};
    r = vkQueuePresentKHR_(W.q, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) W.resized = 1;
    else if (r) LOG("[RED] [vk] vkQueuePresentKHR failed (%d)", r);
    W.frames++;
    if (W.frames == 1) LOG("[vk] first frame in the window");
    if (W.frames % 60 == 0) {
        double t = now_s(); static double last; static uint64_t lastf;
        if (last) { char title[200]; snprintf(title, sizeof title, "nxnative - %s - %.1f fps - frame %llu (graphics: stage A, no picture yet)", W.devname, (W.frames - lastf) / (t - last), (unsigned long long)n); S.SetWindowTitle(W.win, title); }
        last = t; lastf = W.frames;
    }
}
