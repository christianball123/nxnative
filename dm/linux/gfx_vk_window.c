// Stage A of the Vulkan graphics layer (dm/linux/GRAPHICS.md): a window (SDL3) and a Vulkan swapchain that presents once per game present, at the game's pace.
// SDL3 and the Vulkan loader are opened at run time (dlopen), so the build needs neither's development files: ArmadaOS has libSDL3.so.0 and libvulkan.so.1 (sudo dnf install SDL3 vulkan-loader
// in the toolbox if not). Until stage B shows the game's own image, the window shows a colour that steps with every present, so its pace and the frame count are visible.
// The window is made on the thread of the first present (the game's present thread), and its events are read on that thread at every present.
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
enum { SDL_EVENT_QUIT = 0x100, SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED = 0x207, SDL_EVENT_KEY_DOWN = 0x300, SDL_EVENT_GAMEPAD_ADDED = 0x653, SDL_EVENT_GAMEPAD_REMOVED = 0x654 };
typedef struct SDL_Gamepad SDL_Gamepad;
typedef struct SDL_Joystick SDL_Joystick;
typedef uint32_t SDL_JoystickID;
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
    SDL_JoystickID *(*GetGamepads)(int *);
    SDL_Gamepad *(*OpenGamepad)(SDL_JoystickID);
    void (*CloseGamepad)(SDL_Gamepad *);
    bool (*GamepadConnected)(SDL_Gamepad *);
    bool (*GetGamepadButton)(SDL_Gamepad *, int);
    int16_t (*GetGamepadAxis)(SDL_Gamepad *, int);
    const char *(*GetGamepadName)(SDL_Gamepad *);
    const bool *(*GetKeyboardState)(int *);
    void (*free)(void *);
    bool (*SetHint)(const char *, const char *);
    int (*GetGamepadType)(SDL_Gamepad *);
    SDL_JoystickID *(*GetJoysticks)(int *);
    bool (*IsGamepad)(SDL_JoystickID);
    SDL_Joystick *(*OpenJoystick)(SDL_JoystickID);
    void (*CloseJoystick)(SDL_Joystick *);
    bool (*GetJoystickButton)(SDL_Joystick *, int);
    int16_t (*GetJoystickAxis)(SDL_Joystick *, int);
    uint8_t (*GetJoystickHat)(SDL_Joystick *, int);
    const char *(*GetJoystickName)(SDL_Joystick *);
    int (*GetNumJoystickAxes)(SDL_Joystick *);
} S;

static int sdl_load(void) {
    const char *names[] = {"libSDL3.so.0", "libSDL3.so", NULL};
    for (int i = 0; names[i] && !S.lib; i++) S.lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!S.lib) { LOG("[RED] [vk] window: SDL3 is not installed (libSDL3.so.0: %s). In the toolbox: sudo dnf install -y SDL3", dlerror()); return 0; }
#define L(f) if (!(*(void **)&S.f = dlsym(S.lib, "SDL_" #f))) { LOG("[RED] [vk] window: SDL3 has no SDL_" #f); return 0; }
    L(Init) L(GetError) L(CreateWindow) L(SetWindowTitle) L(GetWindowSizeInPixels) L(PollEvent) L(Vulkan_GetInstanceExtensions) L(Vulkan_CreateSurface) L(Vulkan_GetVkGetInstanceProcAddr) L(Vulkan_LoadLibrary)
#undef L
    // the gamepad and keyboard functions: optional (no input without them, the game still runs)
#define LO(f) *(void **)&S.f = dlsym(S.lib, "SDL_" #f);
    LO(GetGamepads) LO(OpenGamepad) LO(CloseGamepad) LO(GamepadConnected) LO(GetGamepadButton) LO(GetGamepadAxis) LO(GetGamepadName) LO(GetKeyboardState) LO(free)
    LO(SetHint) LO(GetGamepadType) LO(GetJoysticks) LO(IsGamepad) LO(OpenJoystick) LO(CloseJoystick) LO(GetJoystickButton) LO(GetJoystickAxis) LO(GetJoystickHat) LO(GetJoystickName) LO(GetNumJoystickAxes)
    // controllers keep working while another window (the terminal) has the focus: in desktop mode the game window is rarely the focused one
    if (S.SetHint) S.SetHint("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
#undef LO
    return 1;
}

// ---- Vulkan functions (the lists are in gfx_vk.h), loaded through SDL's vkGetInstanceProcAddr
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;
static PFN_vkCreateInstance vkCreateInstance_;
#define DEF(f) PFN_##f f##_;
VKI_FUNCS(DEF) VKD_FUNCS(DEF)
#undef DEF
vk_core VK = {.qmu = PTHREAD_MUTEX_INITIALIZER};

#define SWAP_MAX 8
static struct {
    SDL_Window *win;
    VkSurfaceKHR surf;
    VkSwapchainKHR sc; VkFormat fmt; VkExtent2D ext; uint32_t nimg; VkImage img[SWAP_MAX];
    VkSemaphore acquired[SWAP_MAX], done[SWAP_MAX]; VkCommandPool pool; VkCommandBuffer cb[SWAP_MAX]; VkFence fence[SWAP_MAX]; uint32_t slot;
    int ok, failed, resized; uint64_t frames; double t0; char devname[256]; int ext_host_mem, ext_push, ext_dcc;
} W;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static int swapchain_make(void) {
    VkSurfaceCapabilitiesKHR cap; vkGetPhysicalDeviceSurfaceCapabilitiesKHR_(VK.phys, W.surf, &cap);
    int pw = 0, ph = 0; S.GetWindowSizeInPixels(W.win, &pw, &ph);
    VkExtent2D e = cap.currentExtent.width != 0xffffffffu ? cap.currentExtent : (VkExtent2D){(uint32_t)pw, (uint32_t)ph};
    if (!e.width || !e.height) return 0;   // minimized
    uint32_t n = 0; vkGetPhysicalDeviceSurfaceFormatsKHR_(VK.phys, W.surf, &n, NULL);
    VkSurfaceFormatKHR f[64]; if (n > 64) n = 64; vkGetPhysicalDeviceSurfaceFormatsKHR_(VK.phys, W.surf, &n, f);
    VkSurfaceFormatKHR pick = f[0];
    for (uint32_t i = 0; i < n; i++) if (f[i].format == VK_FORMAT_B8G8R8A8_UNORM || f[i].format == VK_FORMAT_R8G8B8A8_UNORM) { pick = f[i]; break; }
    uint32_t count = cap.minImageCount + 1; if (cap.maxImageCount && count > cap.maxImageCount) count = cap.maxImageCount; if (count > SWAP_MAX) count = SWAP_MAX;
    VkSwapchainKHR old = W.sc;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, W.surf, count, pick.format, pick.colorSpace, e, 1,
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
                                   cap.currentTransform, VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_TRUE, old};
    if (!(cap.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) ci.compositeAlpha = (VkCompositeAlphaFlagBitsKHR)(cap.supportedCompositeAlpha & -cap.supportedCompositeAlpha);
    VkResult r = vkCreateSwapchainKHR_(VK.dev, &ci, NULL, &W.sc);
    if (old) { vkDeviceWaitIdle_(VK.dev); vkDestroySwapchainKHR_(VK.dev, old, NULL); }
    if (r) { LOG("[RED] [vk] vkCreateSwapchainKHR %dx%d failed (%d)", e.width, e.height, r); W.sc = VK_NULL_HANDLE; return 0; }
    W.fmt = pick.format; W.ext = e; W.nimg = SWAP_MAX; vkGetSwapchainImagesKHR_(VK.dev, W.sc, &W.nimg, W.img);
    LOG("[vk] swapchain %ux%u, %u images, format %d, FIFO (vsync)", e.width, e.height, W.nimg, pick.format);
    return 1;
}

// Validation messages (DM_VK_VALIDATE): each kind (message ID) logged 3 times at most, so a per-draw mistake does not flood the log
static VKAPI_ATTR VkBool32 VKAPI_CALL val_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
    (void)type; (void)u;
    static int32_t ids[512]; static int cnt[512], n; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    int k = 0; for (; k < n; k++) if (ids[k] == d->messageIdNumber) break;
    if (k == n && n < 512) { ids[n] = d->messageIdNumber; cnt[n++] = 0; }
    int show = k < 512 && cnt[k]++ < 3;
    pthread_mutex_unlock(&mu);
    if (show) LOG("%s[vkval] %s", sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "[RED] " : "", d->pMessage ? d->pMessage : "?");
    return VK_FALSE;
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
    uint32_t next = 0; const char *const *sexts = S.Vulkan_GetInstanceExtensions(&next);
    // DM_VK_VALIDATE=1: the Khronos validation layer (Fedora: sudo dnf install -y vulkan-validation-layers), its messages in the log
    const char *val = getenv("DM_VK_VALIDATE"); int validate = val && *val == '1';
    const char *exts[32]; uint32_t ne0 = 0; for (uint32_t i = 0; i < next && ne0 < 31; i++) exts[ne0++] = sexts[i];
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    if (validate) exts[ne0++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative", 1, "nxnative", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, validate ? 1 : 0, validate ? layers : NULL, ne0, exts};
    VkResult r = vkCreateInstance_(&ici, NULL, &VK.inst);
    if (r && validate) { LOG("[RED] [vk] DM_VK_VALIDATE: the validation layer did not load (%d; sudo dnf install -y vulkan-validation-layers): going on without it", r); validate = 0; ici.enabledLayerCount = 0; ici.enabledExtensionCount = ne0 - 1; r = vkCreateInstance_(&ici, NULL, &VK.inst); }
    if (r) { LOG("[RED] [vk] vkCreateInstance failed (%d)", r); return 0; }
#define LI(f) f##_ = (PFN_##f)vkGetInstanceProcAddr_(VK.inst, #f); if (!f##_) { LOG("[RED] [vk] no " #f); return 0; }
    VKI_FUNCS(LI)
#undef LI
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT cm = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr_(VK.inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, val_cb, NULL};
        VkDebugUtilsMessengerEXT m; if (cm && !cm(VK.inst, &mi, NULL, &m)) LOG("[vk] validation layer on (DM_VK_VALIDATE): its messages follow as [vkval]");
    }
    if (!S.Vulkan_CreateSurface(W.win, VK.inst, NULL, &W.surf)) { LOG("[RED] [vk] SDL_Vulkan_CreateSurface: %s", S.GetError()); return 0; }
    // A real GPU before a software one (llvmpipe, lavapipe: VK_PHYSICAL_DEVICE_TYPE_CPU), which Mesa also installs; DM_VK_GPU=<n> picks by its index in vulkaninfo's list
    uint32_t np = 8; VkPhysicalDevice pd[8]; vkEnumeratePhysicalDevices_(VK.inst, &np, pd);
    const char *pick = getenv("DM_VK_GPU");
    for (int pass = 0; pass < 2 && !VK.phys; pass++)
        for (uint32_t i = 0; i < np && !VK.phys; i++) {
            VkPhysicalDeviceProperties dp; vkGetPhysicalDeviceProperties_(pd[i], &dp);
            if (pick && *pick ? (int)i != atoi(pick) : pass == 0 && dp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            uint32_t nq = 16; VkQueueFamilyProperties qp[16]; vkGetPhysicalDeviceQueueFamilyProperties_(pd[i], &nq, qp);
            for (uint32_t k = 0; k < nq; k++) {
                VkBool32 pres = 0; vkGetPhysicalDeviceSurfaceSupportKHR_(pd[i], k, W.surf, &pres);
                if ((qp[k].queueFlags & VK_QUEUE_GRAPHICS_BIT) && pres) { VK.phys = pd[i]; VK.qfam = k; break; }
            }
        }
    if (!VK.phys) { LOG("[RED] [vk] no GPU can draw to this window"); return 0; }
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties_(VK.phys, &pp);
    snprintf(W.devname, sizeof W.devname, "%s", pp.deviceName);
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties_(VK.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep); vkEnumerateDeviceExtensionProperties_(VK.phys, NULL, &ne, ep);
    for (uint32_t i = 0; i < ne; i++) {
        if (!strcmp(ep[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) W.ext_host_mem = 1;
        if (!strcmp(ep[i].extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) W.ext_push = 1;
        if (!strcmp(ep[i].extensionName, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME)) W.ext_dcc = 1;
    }
    free(ep);
    LOG("[vk] GPU %s, Vulkan %u.%u.%u, driver %#x; guest memory import (VK_EXT_external_memory_host): %s", pp.deviceName, VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion),
        VK_API_VERSION_PATCH(pp.apiVersion), pp.driverVersion, W.ext_host_mem ? "yes" : "no (stage B will mirror pools instead)");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, VK.qfam, 1, &prio};
    // Stage C's shaders and pipelines: Vulkan 1.3 (dynamic rendering, extended dynamic state) with every feature the GPU has turned on (the translated shaders use what the game's do:
    // image queries, gathers with offsets, subgroup ballots, texel buffers, storage image access without a format, …), and push descriptors when the driver has them
    const char *dext[4] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME}; uint32_t ndext = 1;
    if (VK_API_VERSION_MINOR(pp.apiVersion) < 3 && VK_API_VERSION_MAJOR(pp.apiVersion) == 1) LOG("[RED] [vk] this driver has Vulkan %u.%u: stage C needs 1.3", VK_API_VERSION_MAJOR(pp.apiVersion), VK_API_VERSION_MINOR(pp.apiVersion));
    if (W.ext_push) { dext[ndext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME; VK.push_desc = 1; }
    static VkPhysicalDeviceDepthClipControlFeaturesEXT fdcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    static VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    if (W.ext_dcc) { dext[ndext++] = VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME; f13.pNext = &fdcc; }
    static VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    static VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    static VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    PFN_vkGetPhysicalDeviceFeatures2 gpf2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr_(VK.inst, "vkGetPhysicalDeviceFeatures2");
    if (gpf2) gpf2(VK.phys, &f2);
    f2.features.robustBufferAccess = 0;   // costs speed and the game does not ask for it
    VK.f_dynrender = f13.dynamicRendering; VK.depth_clip_control = W.ext_dcc && fdcc.depthClipControl;
    VK.f_logic_op = f2.features.logicOp; VK.f_dual_src = f2.features.dualSrcBlend; VK.f_depth_clamp = f2.features.depthClamp; VK.f_wide_lines = f2.features.wideLines; VK.limits = pp.limits;
    LOG("[vk] features: dynamic rendering %d, GL depth clipping %d, logic ops %d, dual-source blending %d, depth clamp %d, push descriptors %d; BC textures %d",
        VK.f_dynrender, VK.depth_clip_control, VK.f_logic_op, VK.f_dual_src, VK.f_depth_clamp, W.ext_push, f2.features.textureCompressionBC);
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, gpf2 ? &f2 : NULL, 0, 1, &qci, 0, NULL, ndext, dext, NULL};
    if ((r = vkCreateDevice_(VK.phys, &dci, NULL, &VK.dev))) { LOG("[RED] [vk] vkCreateDevice failed (%d)", r); return 0; }
#define LD(f) f##_ = (PFN_##f)vkGetDeviceProcAddr_(VK.dev, #f); if (!f##_) { LOG("[RED] [vk] no " #f); return 0; }
    VKD_FUNCS(LD)
#undef LD
    vkGetDeviceQueue_(VK.dev, VK.qfam, 0, &VK.q);
    vkGetPhysicalDeviceMemoryProperties_(VK.phys, &VK.mem);
    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VK.qfam};
    vkCreateCommandPool_(VK.dev, &pci, NULL, &W.pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, W.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, SWAP_MAX};
    vkAllocateCommandBuffers_(VK.dev, &cai, W.cb);
    for (int i = 0; i < SWAP_MAX; i++) {
        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, NULL, 0};
        vkCreateSemaphore_(VK.dev, &si, NULL, &W.acquired[i]); vkCreateSemaphore_(VK.dev, &si, NULL, &W.done[i]);
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT};
        vkCreateFence_(VK.dev, &fi, NULL, &W.fence[i]);
    }
    if (!swapchain_make()) return 0;
    W.t0 = now_s();
    return 1;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier_(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}

// ---- input (stage E): SDL3 gamepads → the game's controllers (hle_hid.c dm_pad_set_player), player n = the n-th gamepad; the keyboard (window focused) adds to player 1.
// Face buttons by position, as on a Switch: bottom B, right A, left Y, top X (the Thor's buttons are labelled that way). DM_PAD_LAYOUT=xbox maps by an Xbox pad's labels
// instead (A at the bottom). Shoulders L / R, triggers ZL / ZR, Start +, Back -, stick clicks; the sticks with up positive, as the Switch reports them.
// Keyboard: arrows = d-pad, X = A, Z = B, S = X, A = Y, Q = L, W = R, 1 = ZL, 2 = ZR, Enter = +, Backspace = -, I J K L = left stick
#include "../host/dm_input.h"
enum { GP_SOUTH, GP_EAST, GP_WEST, GP_NORTH, GP_BACK, GP_GUIDE, GP_START, GP_LSTICK, GP_RSTICK, GP_LSHOULDER, GP_RSHOULDER, GP_UP, GP_DOWN, GP_LEFT, GP_RIGHT };
enum { GA_LX, GA_LY, GA_RX, GA_RY, GA_LT, GA_RT };
// A joystick SDL has no gamepad mapping for is read as an Xbox 360 pad (the Linux xpad driver's order: buttons A B X Y LB RB Back Start Guide LS RS,
// axes LX LY LT RX RY RT, the d-pad as hat 0), which is how most handhelds' built-in controls and 360-compatible pads report
static SDL_Gamepad *g_pad[DM_PLAYERS]; static SDL_Joystick *g_joy[DM_PLAYERS]; static int g_npad, g_pad_scan = 1, g_xbox = -1;
static const char *k_gptype[] = {"unknown", "standard", "Xbox 360", "Xbox One", "PS3", "PS4", "PS5", "Switch Pro", "Joy-Con (L)", "Joy-Con (R)", "Joy-Con pair"};
static void pads_scan(void) {
    g_pad_scan = 0;
    for (int i = 0; i < g_npad; i++) { if (g_pad[i]) S.CloseGamepad(g_pad[i]); if (g_joy[i] && S.CloseJoystick) S.CloseJoystick(g_joy[i]); g_pad[i] = NULL; g_joy[i] = NULL; }
    g_npad = 0;
    int n = 0; SDL_JoystickID *ids = S.GetGamepads ? S.GetGamepads(&n) : NULL;
    for (int i = 0; i < n && g_npad < DM_PLAYERS; i++) {
        SDL_Gamepad *g = S.OpenGamepad(ids[i]); if (!g) continue;
        int t = S.GetGamepadType ? S.GetGamepadType(g) : 0;
        g_pad[g_npad++] = g;
        LOG("[input] player %d: %s (%s pad)", g_npad, S.GetGamepadName ? S.GetGamepadName(g) : "gamepad", t >= 0 && t < 11 ? k_gptype[t] : "other");
    }
    if (ids && S.free) S.free(ids);
    ids = S.GetJoysticks && S.IsGamepad && S.OpenJoystick ? S.GetJoysticks(&n) : NULL;
    for (int i = 0; ids && i < n && g_npad < DM_PLAYERS; i++) {
        if (S.IsGamepad(ids[i])) continue;
        SDL_Joystick *j = S.OpenJoystick(ids[i]); if (!j) continue;
        g_joy[g_npad++] = j;
        LOG("[input] player %d: %s (no gamepad mapping: read as an Xbox 360 pad)", g_npad, S.GetJoystickName ? S.GetJoystickName(j) : "joystick");
    }
    if (ids && S.free) S.free(ids);
    if (!g_npad) LOG("[input] no gamepad found: the keyboard plays player 1 (arrows, X = A, Z = B, S = X, A = Y, Q / W = L / R, 1 / 2 = ZL / ZR, Enter = +, Backspace = -)");
    dm_pad_players(g_npad > 1 ? (1u << g_npad) - 1 : 1);
}
static int32_t stick(int16_t v) { return v < -DM_STICK_MAX ? -DM_STICK_MAX : v; }
static void input_poll(void) {
    if (!S.GetGamepadButton) return;
    if (g_xbox < 0) { const char *e = getenv("DM_PAD_LAYOUT"); g_xbox = e && !strcmp(e, "xbox"); }
    if (g_pad_scan) pads_scan();
    for (int p = 0; p < g_npad; p++) {
        SDL_Joystick *j = g_joy[p];
        if (j) {   // a raw joystick in the xpad layout
            #define JB(x) S.GetJoystickButton(j, x)
            uint64_t b = 0; uint8_t hat = S.GetJoystickHat ? S.GetJoystickHat(j, 0) : 0; int na = S.GetNumJoystickAxes ? S.GetNumJoystickAxes(j) : 6;
            if (JB(g_xbox ? 0 : 1)) b |= DM_BTN_A;
            if (JB(g_xbox ? 1 : 0)) b |= DM_BTN_B;
            if (JB(g_xbox ? 3 : 2)) b |= DM_BTN_Y;
            if (JB(g_xbox ? 2 : 3)) b |= DM_BTN_X;
            if (JB(4)) b |= DM_BTN_L;
            if (JB(5)) b |= DM_BTN_R;
            if (JB(6)) b |= DM_BTN_MINUS;
            if (JB(7)) b |= DM_BTN_PLUS;
            if (JB(9)) b |= DM_BTN_LS;
            if (JB(10)) b |= DM_BTN_RS;
            #undef JB
            if (hat & 1) b |= DM_BTN_UP;
            if (hat & 2) b |= DM_BTN_RIGHT;
            if (hat & 4) b |= DM_BTN_DOWN;
            if (hat & 8) b |= DM_BTN_LEFT;
            if (na >= 6) { if (S.GetJoystickAxis(j, 2) > 0) b |= DM_BTN_ZL; if (S.GetJoystickAxis(j, 5) > 0) b |= DM_BTN_ZR; }   // triggers rest at -32768
            int32_t lx = stick(S.GetJoystickAxis(j, 0)), ly = -stick(S.GetJoystickAxis(j, 1));
            int32_t rx = na >= 5 ? stick(S.GetJoystickAxis(j, 3)) : 0, ry = na >= 5 ? -stick(S.GetJoystickAxis(j, 4)) : 0;
            dm_pad_set_player(p, b, lx, ly, rx, ry);
            continue;
        }
        SDL_Gamepad *g = g_pad[p];
        if (!g || (S.GamepadConnected && !S.GamepadConnected(g))) continue;
        #define B(x) S.GetGamepadButton(g, x)
        uint64_t b = 0;
        if (B(g_xbox ? GP_SOUTH : GP_EAST)) b |= DM_BTN_A;
        if (B(g_xbox ? GP_EAST : GP_SOUTH)) b |= DM_BTN_B;
        if (B(g_xbox ? GP_NORTH : GP_WEST)) b |= DM_BTN_Y;
        if (B(g_xbox ? GP_WEST : GP_NORTH)) b |= DM_BTN_X;
        if (B(GP_LSHOULDER)) b |= DM_BTN_L;
        if (B(GP_RSHOULDER)) b |= DM_BTN_R;
        if (B(GP_START)) b |= DM_BTN_PLUS;
        if (B(GP_BACK)) b |= DM_BTN_MINUS;
        if (B(GP_LSTICK)) b |= DM_BTN_LS;
        if (B(GP_RSTICK)) b |= DM_BTN_RS;
        if (B(GP_UP)) b |= DM_BTN_UP;
        if (B(GP_DOWN)) b |= DM_BTN_DOWN;
        if (B(GP_LEFT)) b |= DM_BTN_LEFT;
        if (B(GP_RIGHT)) b |= DM_BTN_RIGHT;
        #undef B
        if (S.GetGamepadAxis(g, GA_LT) > 12000) b |= DM_BTN_ZL;
        if (S.GetGamepadAxis(g, GA_RT) > 12000) b |= DM_BTN_ZR;
        int32_t lx = stick(S.GetGamepadAxis(g, GA_LX)), ly = -stick(S.GetGamepadAxis(g, GA_LY)), rx = stick(S.GetGamepadAxis(g, GA_RX)), ry = -stick(S.GetGamepadAxis(g, GA_RY));
        dm_pad_set_player(p, b, lx, ly, rx, ry);
    }
    // the keyboard: player 1's keyboard path (ORed with its gamepad)
    const bool *k = S.GetKeyboardState ? S.GetKeyboardState(NULL) : NULL;
    if (k) {
        uint64_t b = 0; int32_t lx = 0, ly = 0;
        if (k[27]) b |= DM_BTN_A; if (k[29]) b |= DM_BTN_B; if (k[22]) b |= DM_BTN_X; if (k[4]) b |= DM_BTN_Y;
        if (k[20]) b |= DM_BTN_L; if (k[26]) b |= DM_BTN_R; if (k[30]) b |= DM_BTN_ZL; if (k[31]) b |= DM_BTN_ZR;
        if (k[40]) b |= DM_BTN_PLUS; if (k[42]) b |= DM_BTN_MINUS;
        if (k[82]) b |= DM_BTN_UP; if (k[81]) b |= DM_BTN_DOWN; if (k[80]) b |= DM_BTN_LEFT; if (k[79]) b |= DM_BTN_RIGHT;
        if (k[12]) ly += DM_STICK_MAX; if (k[14]) ly -= DM_STICK_MAX; if (k[13]) lx -= DM_STICK_MAX; if (k[15]) lx += DM_STICK_MAX;   // I K J L
        dm_input_set(b, lx, ly, 0, 0);
    }
}

static void events(void) {
    SDL_Event e;
    while (S.PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT) { LOG("[vk] window closed: quitting"); fflush(NULL); _exit(0); }
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) W.resized = 1;
        if (e.type == SDL_EVENT_GAMEPAD_ADDED || e.type == SDL_EVENT_GAMEPAD_REMOVED) g_pad_scan = 1;
    }
    input_poll();
}

uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < VK.mem.memoryTypeCount; i++) if ((bits & 1u << i) && (VK.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return ~0u;
}
int vkw_active(void) { return W.ok; }
int vkw_wanted(void) { const char *w = getenv("DM_WINDOW"); return w && *w && *w != '0'; }
int vkw_init(void) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (!W.ok && !W.failed) {
        if (window_init()) { W.ok = 1; VK.ok = 1; }
        else { W.failed = 1; LOG("[RED] [vk] no window: running headless"); }
    }
    pthread_mutex_unlock(&mu);
    return W.ok;
}

// One present: the next swapchain image, with the game's image scaled into it (or stage A's colour when there is none yet), shown with FIFO: the display's vsync paces the game
// as a Switch's display would. The game's image stays in VK_IMAGE_LAYOUT_GENERAL, the layout gfx_vk_replay.c keeps every image in
void vkw_present(uint64_t n, VkImage gimg, uint32_t gw, uint32_t gh, const int crop[4]) {
    if (!vkw_init()) return;
    events();
    pthread_mutex_lock(&VK.qmu);
    if (W.resized || !W.sc) { W.resized = 0; if (!swapchain_make()) { pthread_mutex_unlock(&VK.qmu); return; } }
    uint32_t s = W.slot = (W.slot + 1) % W.nimg;
    vkWaitForFences_(VK.dev, 1, &W.fence[s], VK_TRUE, UINT64_MAX);
    uint32_t idx;
    VkResult r = vkAcquireNextImageKHR_(VK.dev, W.sc, UINT64_MAX, W.acquired[s], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.resized = 1; pthread_mutex_unlock(&VK.qmu); return; }
    if (r && r != VK_SUBOPTIMAL_KHR) { LOG("[RED] [vk] vkAcquireNextImageKHR failed (%d)", r); pthread_mutex_unlock(&VK.qmu); return; }
    vkResetFences_(VK.dev, 1, &W.fence[s]);
    VkCommandBuffer cb = W.cb[s]; vkResetCommandBuffer_(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vkBeginCommandBuffer_(cb, &bi);
    barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (gimg) {
        VkClearColorValue black = {{0, 0, 0, 1}};
        vkCmdClearColorImage_(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
        // the game's crop (WindowSetCrop), fitted into the window with its aspect kept
        int cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] > 0 ? crop[2] : (int)gw, ch = crop && crop[3] > 0 ? crop[3] : (int)gh;
        double sx = (double)W.ext.width / cw, sy = (double)W.ext.height / ch, k = sx < sy ? sx : sy;
        int dw = (int)(cw * k), dh = (int)(ch * k), dx = ((int)W.ext.width - dw) / 2, dy = ((int)W.ext.height - dh) / 2;
        VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{cx, cy, 0}, {cx + cw, cy + ch, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier_(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdBlitImage_(cb, gimg, VK_IMAGE_LAYOUT_GENERAL, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    } else {
        // Stage A's picture: a dark blue that brightens and dims once a second at 60 presents a second (so a steady pulse means a steady 60 fps)
        double ph = (double)(n % 60) / 60.0, k = 0.5 - 0.5 * cos(ph * 2 * M_PI);
        VkClearColorValue c = {{0.02f + 0.10f * (float)k, 0.06f + 0.20f * (float)k, 0.20f + 0.45f * (float)k, 1.0f}};
        vkCmdClearColorImage_(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rr);
    }
    barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer_(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &W.acquired[s], &ws, 1, &cb, 1, &W.done[s]};
    if ((r = vkQueueSubmit_(VK.q, 1, &si, W.fence[s]))) { LOG("[RED] [vk] vkQueueSubmit failed (%d)", r); pthread_mutex_unlock(&VK.qmu); return; }
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &W.done[s], 1, &W.sc, &idx, NULL};
    r = vkQueuePresentKHR_(VK.q, &pi);
    pthread_mutex_unlock(&VK.qmu);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) W.resized = 1;
    else if (r) LOG("[RED] [vk] vkQueuePresentKHR failed (%d)", r);
    W.frames++;
    if (W.frames == 1) LOG("[vk] first frame in the window");
    if (W.frames % 60 == 0) {
        double t = now_s(); static double last; static uint64_t lastf;
        if (last) { char title[200]; snprintf(title, sizeof title, "nxnative - %s - %.1f fps - frame %llu", W.devname, (W.frames - lastf) / (t - last), (unsigned long long)n); S.SetWindowTitle(W.win, title); }
        last = t; lastf = W.frames;
    }
}
