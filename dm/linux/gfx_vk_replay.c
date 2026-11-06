// Stage B of the Vulkan graphics layer (dm/linux/GRAPHICS.md): the replay of the game's recorded NVN commands (nvn_obj.h records, as dm/gfx/nvn/nvn_mtl.m replays them on Metal).
//
// What it does now:
//   - the CPU-visible effects, with or without a window: events (SignalEvent), counters (ReportCounter), fence syncs, buffer copies, fills and inline uniform updates. Buffers are the
//     guest's own memory, which the CPU reads and writes directly, so these run on the CPU in record order; the effects the game waits for (events, counters) land when the submission
//     completes, as on the GPU
//   - with a window (DM_WINDOW=1): textures as VkImages (by serial, made at first use), colour and depth / stencil clears of the render targets, texture clears, buffer → texture,
//     texture → texture and texture → buffer copies, and the present of the game's swapchain image
// What it does not do yet: draws and compute (stage C: shaders), packaged texture uploads (their storage is block-linear), partial clears (a scissor, a channel mask), sparse textures.
// Every image stays in VK_IMAGE_LAYOUT_GENERAL, and every GPU operation is followed by a full memory barrier: simple and correct, not fast. A submission waits for the GPU to finish.
#include "dm.h"
#include "../gfx/nvn/nvn_obj.h"
#include "../gfx/nvn/nvn_mtl.h"
#include "../gfx/nvn/nvn_timestamp.h"
#include "../gfx/tex/nvn_fmt_mtl.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include "gfx_vk.h"
#include <stdatomic.h>

uint64_t guest_ns(void);

// ---------------------------------------------------------------- formats: NVN → (Metal, as dm/gfx/tex/nvn_fmt_mtl.c maps it) → Vulkan
static VkFormat vk_of_mtl(uint32_t m) {
    switch (m) {
    case 10: return VK_FORMAT_R8_UNORM; case 12: return VK_FORMAT_R8_SNORM; case 13: return VK_FORMAT_R8_UINT; case 14: return VK_FORMAT_R8_SINT;
    case 20: return VK_FORMAT_R16_UNORM; case 22: return VK_FORMAT_R16_SNORM; case 23: return VK_FORMAT_R16_UINT; case 24: return VK_FORMAT_R16_SINT; case 25: return VK_FORMAT_R16_SFLOAT;
    case 30: return VK_FORMAT_R8G8_UNORM; case 32: return VK_FORMAT_R8G8_SNORM; case 33: return VK_FORMAT_R8G8_UINT; case 34: return VK_FORMAT_R8G8_SINT;
    case 40: return VK_FORMAT_R5G6B5_UNORM_PACK16; case 41: return VK_FORMAT_A1R5G5B5_UNORM_PACK16; case 42: return VK_FORMAT_R4G4B4A4_UNORM_PACK16; case 43: return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
    case 53: return VK_FORMAT_R32_UINT; case 54: return VK_FORMAT_R32_SINT; case 55: return VK_FORMAT_R32_SFLOAT;
    case 60: return VK_FORMAT_R16G16_UNORM; case 62: return VK_FORMAT_R16G16_SNORM; case 63: return VK_FORMAT_R16G16_UINT; case 64: return VK_FORMAT_R16G16_SINT; case 65: return VK_FORMAT_R16G16_SFLOAT;
    case 70: return VK_FORMAT_R8G8B8A8_UNORM; case 71: return VK_FORMAT_R8G8B8A8_SRGB; case 72: return VK_FORMAT_R8G8B8A8_SNORM; case 73: return VK_FORMAT_R8G8B8A8_UINT; case 74: return VK_FORMAT_R8G8B8A8_SINT;
    case 80: return VK_FORMAT_B8G8R8A8_UNORM; case 81: return VK_FORMAT_B8G8R8A8_SRGB;
    case 90: return VK_FORMAT_A2B10G10R10_UNORM_PACK32; case 91: return VK_FORMAT_A2B10G10R10_UINT_PACK32; case 92: return VK_FORMAT_B10G11R11_UFLOAT_PACK32; case 93: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 103: return VK_FORMAT_R32G32_UINT; case 104: return VK_FORMAT_R32G32_SINT; case 105: return VK_FORMAT_R32G32_SFLOAT;
    case 110: return VK_FORMAT_R16G16B16A16_UNORM; case 112: return VK_FORMAT_R16G16B16A16_SNORM; case 113: return VK_FORMAT_R16G16B16A16_UINT; case 114: return VK_FORMAT_R16G16B16A16_SINT; case 115: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 123: return VK_FORMAT_R32G32B32A32_UINT; case 124: return VK_FORMAT_R32G32B32A32_SINT; case 125: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 130: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK; case 131: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK; case 132: return VK_FORMAT_BC2_UNORM_BLOCK; case 133: return VK_FORMAT_BC2_SRGB_BLOCK;
    case 134: return VK_FORMAT_BC3_UNORM_BLOCK; case 135: return VK_FORMAT_BC3_SRGB_BLOCK; case 140: return VK_FORMAT_BC4_UNORM_BLOCK; case 141: return VK_FORMAT_BC4_SNORM_BLOCK;
    case 142: return VK_FORMAT_BC5_UNORM_BLOCK; case 143: return VK_FORMAT_BC5_SNORM_BLOCK; case 150: return VK_FORMAT_BC6H_SFLOAT_BLOCK; case 151: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case 152: return VK_FORMAT_BC7_UNORM_BLOCK; case 153: return VK_FORMAT_BC7_SRGB_BLOCK;
    case 250: return VK_FORMAT_D16_UNORM; case 252: return VK_FORMAT_D32_SFLOAT; case 253: return VK_FORMAT_S8_UINT; case 260: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    }
    // ASTC: Metal's LDR from 204, sRGB from 186, 4x4 … 12x12 with one number skipped after 6x6 (nvn_fmt_mtl.c astc_off); Vulkan's are consecutive in the same order, UNORM then SRGB
    static const uint8_t back[15] = {0, 1, 2, 3, 4, 0xff, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    if (m >= 204 && m < 219 && back[m - 204] != 0xff) return (VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * back[m - 204]);
    if (m >= 186 && m < 201 && back[m - 186] != 0xff) return (VkFormat)(VK_FORMAT_ASTC_4x4_SRGB_BLOCK + 2 * back[m - 186]);
    return VK_FORMAT_UNDEFINED;
}
static int fmt_depth(VkFormat f) { return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT; }
static VkImageAspectFlags fmt_aspects(VkFormat f) {
    switch (f) { case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT: return VK_IMAGE_ASPECT_DEPTH_BIT; case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
                 case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT; default: return VK_IMAGE_ASPECT_COLOR_BIT; }
}
static int fmt_int(VkFormat f) {   // integer formats clear with integer values
    switch (f) { case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R8G8_UINT:
                 case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R8G8B8A8_UINT:
                 case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
                 case VK_FORMAT_A2B10G10R10_UINT_PACK32: return 1; default: return 0; }
}

// ---------------------------------------------------------------- textures: one VkImage per texture serial
struct vview;
// gpu: the GPU has written it (render target, clear, copy destination, shader image): its content lives here and is never replaced from guest memory;
// uploaded: its guest-memory content was copied in (stage C uploads a texture the GPU has not written at its first sampling, and again after the CPU writes it: mtl_tex_cpu_written)
typedef struct { uint64_t serial; VkImage img; VkDeviceMemory mem; VkFormat fmt; uint32_t w, h, d, layers, levels, samples; int is3d, is1d, bad, gpu, uploaded, storage; struct vview *views; } vtex;
#define TEX_CAP (1u << 16)
static vtex g_tex[TEX_CAP];   // open addressing by serial
static pthread_mutex_t g_tex_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_tex_made, g_tex_bytes;
static vtex *tex_slot(uint64_t serial, int make) {
    uint32_t i = (uint32_t)(serial * 0x9E3779B97F4A7C15ull >> 48) & (TEX_CAP - 1);
    for (uint32_t k = 0; k < TEX_CAP; k++, i = (i + 1) & (TEX_CAP - 1)) {
        if (g_tex[i].serial == serial) return &g_tex[i];
        if (!g_tex[i].serial || g_tex[i].serial == ~0ull) { if (!make) return NULL; if (!g_tex[i].serial) { memset(&g_tex[i], 0, sizeof g_tex[i]); g_tex[i].serial = serial; return &g_tex[i]; } }
    }
    return NULL;
}
static vtex *tex_get(const tex_t *t) {
    if (!t || !t->serial || !VK.ok) return NULL;
    pthread_mutex_lock(&g_tex_mu);
    vtex *v = tex_slot(t->serial, 1);
    if (!v || v->img || v->bad) { pthread_mutex_unlock(&g_tex_mu); return v && v->img ? v : NULL; }
    uint32_t mf = 0; uint8_t swz[4]; nvn_fmt_to_mtl(t->d.format, &mf, swz);
    VkFormat f = vk_of_mtl(mf);
    int target = t->d.target;
    if (!f || target == NVN_TEXTURE_TARGET_BUFFER) { v->bad = 1; pthread_mutex_unlock(&g_tex_mu); static int once; if (!once++) LOG("[vk] texture format %#x / target %d has no Vulkan image yet (first one: serial %llu)", t->d.format, target, (unsigned long long)t->serial); return NULL; }
    int depth = fmt_depth(f);
    uint32_t samples = t->d.samples > 1 ? (uint32_t)t->d.samples : 1;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f, {t->d.width ? t->d.width : 1, t->d.height ? t->d.height : 1, 1}, t->d.levels ? t->d.levels : 1, 1,
                            (VkSampleCountFlagBits)samples, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT), VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: ci.imageType = VK_IMAGE_TYPE_1D; ci.extent.height = 1; break;   // a real 1D image: the translated shaders sample it as one
    case NVN_TEXTURE_TARGET_3D: ci.imageType = VK_IMAGE_TYPE_3D; ci.extent.depth = t->d.depth ? t->d.depth : 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: ci.imageType = VK_IMAGE_TYPE_1D; ci.arrayLayers = t->d.height ? t->d.height : 1; ci.extent.height = 1; break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: ci.arrayLayers = t->d.depth ? t->d.depth : 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP: ci.arrayLayers = 6; ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: ci.arrayLayers = t->d.depth ? t->d.depth : 6; ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    default: break;
    }
    // Usage the format allows: drop attachment / sampling where the device has no support (block-compressed formats are never attachments)
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties_(VK.phys, f, &fp); VkFormatFeatureFlags ff = fp.optimalTilingFeatures;
    if (!(ff & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))) ci.usage &= ~(VkImageUsageFlags)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    if (!(ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) ci.usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1) { ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT; v->storage = 1; }   // shader images (nvnCommandBufferBindImage)
    if (samples > 1) ci.mipLevels = 1;
    if (!ff) { v->bad = 1; pthread_mutex_unlock(&g_tex_mu); static int once2; if (!once2++) LOG("[RED] [vk] the GPU has no support for Vulkan format %d (NVN %#x): such textures are skipped (BC formats on Adreno need a CPU decode)", f, t->d.format); return NULL; }
    VkResult r = vkCreateImage_(VK.dev, &ci, NULL, &v->img);
    if (r) { v->bad = 1; v->img = VK_NULL_HANDLE; pthread_mutex_unlock(&g_tex_mu); LOG("[RED] [vk] vkCreateImage %ux%ux%u x%u levels %u format %d failed (%d)", ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, f, r); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements_(VK.dev, v->img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    if (ai.memoryTypeIndex == ~0u) ai.memoryTypeIndex = vk_mem_type(mr.memoryTypeBits, 0);
    if ((r = vkAllocateMemory_(VK.dev, &ai, NULL, &v->mem)) || (r = vkBindImageMemory_(VK.dev, v->img, v->mem, 0))) {
        LOG("[RED] [vk] texture memory (%.1f MB) failed (%d)", mr.size / 1048576.0, r); vkDestroyImage_(VK.dev, v->img, NULL); v->img = VK_NULL_HANDLE; v->bad = 1; pthread_mutex_unlock(&g_tex_mu); return NULL; }
    v->fmt = f; v->w = ci.extent.width; v->h = ci.extent.height; v->d = ci.extent.depth; v->layers = ci.arrayLayers; v->levels = ci.mipLevels; v->is3d = ci.imageType == VK_IMAGE_TYPE_3D; v->is1d = ci.imageType == VK_IMAGE_TYPE_1D; v->samples = samples;
    g_tex_made++; g_tex_bytes += mr.size;
    if (g_tex_made == 1 || g_tex_made % 500 == 0) LOG("[vk] textures made: %llu, %.0f MB", (unsigned long long)g_tex_made, g_tex_bytes / 1048576.0);
    pthread_mutex_unlock(&g_tex_mu);
    return v;
}
static void vk_views_drop(vtex *v);
void mtl_tex_init(const tex_t *t) { (void)t; }   // made at first use (tex_get): most textures a game initializes are packaged textures stage B does not upload yet
void mtl_tex_release(tex_t *t) {
    if (!t || !t->serial) return;
    pthread_mutex_lock(&g_tex_mu);
    vtex *v = tex_slot(t->serial, 0);
    if (v) {
        if (v->img) { pthread_mutex_lock(&VK.qmu); vkQueueWaitIdle_(VK.q); pthread_mutex_unlock(&VK.qmu); vk_views_drop(v); vkDestroyImage_(VK.dev, v->img, NULL); vkFreeMemory_(VK.dev, v->mem, NULL); }
        memset(v, 0, sizeof *v); v->serial = ~0ull;   // a tombstone keeps later probes going
    }
    pthread_mutex_unlock(&g_tex_mu);
}

// ---------------------------------------------------------------- the submission: one command buffer, waited for at commit
static VkCommandPool g_pool; static VkCommandBuffer g_cb; static VkFence g_fence_vk; static int g_cb_open, g_cb_ops;
static _Atomic uint64_t g_fence;   // submissions committed (each completes before the next starts)
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // 1 event, 2 counter
static post_op *g_post; static int g_npost, g_postcap;
static struct { p_set_rt rt; int has_rt; int64_t copy_stride, copy_img_stride; } R;
// Staging buffers of this submission (texture uploads and readbacks), freed at commit; readbacks copy into guest memory after the GPU finishes
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint64_t n; uint8_t *rb_dst; size_t rb_row, rb_src_row, rb_rows, rb_n; } stage_t;
static stage_t *g_stage; static int g_nstage, g_stagecap;

static void post(post_op o) { if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); } g_post[g_npost++] = o; }
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; } return nvn_timestamp_from_ns(g); }

static int cb_begin(void) {
    if (!VK.ok) return 0;
    if (!g_pool) {
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VK.qfam};
        vkCreateCommandPool_(VK.dev, &pci, NULL, &g_pool);
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkAllocateCommandBuffers_(VK.dev, &cai, &g_cb);
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0}; vkCreateFence_(VK.dev, &fi, NULL, &g_fence_vk);
    }
    if (!g_cb_open) {
        vkResetCommandBuffer_(g_cb, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer_(g_cb, &bi); g_cb_open = 1; g_cb_ops = 0;
    }
    return 1;
}
static void full_barrier(void) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};
    vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
// An image first used: from UNDEFINED to GENERAL, cleared to zero (new NVN textures start as zero). vtex.bad bit 1 marks it done
static int img_general(vtex *v) {
    if (v->bad & 2) return 1;
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, v->img, {fmt_aspects(v->fmt), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkImageSubresourceRange rr = {fmt_aspects(v->fmt), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (fmt_depth(v->fmt)) { VkClearDepthStencilValue z = {0, 0}; vkCmdClearDepthStencilImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    else if (!(v->fmt >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && v->fmt <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)) { VkClearColorValue z = {{0}}; vkCmdClearColorImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    full_barrier();
    v->bad |= 2; return 1;
}
static vtex *use(const tex_t *t) { vtex *v = tex_get(t); if (!v || !cb_begin()) return NULL; img_general(v); g_cb_ops++; return v; }
static void view_range(const vtex *v, const view_t *vw, int has, uint32_t *lv, uint32_t *nl, uint32_t *l0, uint32_t *nlay) {
    *lv = has && vw->set_levels ? vw->base_level : 0; *nl = has && vw->set_levels && vw->nlevels ? vw->nlevels : v->levels - *lv;
    *l0 = has && vw->set_layers ? vw->min_layer : 0; *nlay = has && vw->set_layers && vw->nlayers ? vw->nlayers : (v->is3d ? 1 : v->layers - *l0);
    if (*lv >= v->levels) *lv = v->levels - 1;
    if (*lv + *nl > v->levels) *nl = v->levels - *lv;
    if (*l0 >= v->layers) *l0 = 0;
    if (*l0 + *nlay > v->layers) *nlay = v->layers - *l0;
}

static stage_t *stage_new(uint64_t n) {
    if (g_nstage == g_stagecap) { g_stagecap = g_stagecap ? g_stagecap * 2 : 16; g_stage = realloc(g_stage, g_stagecap * sizeof *g_stage); }
    stage_t *s = &g_stage[g_nstage]; memset(s, 0, sizeof *s);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n ? n : 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer_(VK.dev, &bci, NULL, &s->b)) return NULL;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements_(VK.dev, s->b, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    if (ai.memoryTypeIndex == ~0u || vkAllocateMemory_(VK.dev, &ai, NULL, &s->m)) { vkDestroyBuffer_(VK.dev, s->b, NULL); return NULL; }
    vkBindBufferMemory_(VK.dev, s->b, s->m, 0); vkMapMemory_(VK.dev, s->m, 0, VK_WHOLE_SIZE, 0, &s->p); s->n = n;
    g_nstage++; return s;
}
static void stages_done(void) {
    for (int i = 0; i < g_nstage; i++) {
        stage_t *s = &g_stage[i];
        if (s->rb_dst) for (size_t r = 0; r < s->rb_rows; r++) memcpy(s->rb_dst + r * s->rb_row, (uint8_t *)s->p + r * s->rb_src_row, s->rb_n);
        vkUnmapMemory_(VK.dev, s->m); vkDestroyBuffer_(VK.dev, s->b, NULL); vkFreeMemory_(VK.dev, s->m, NULL);
    }
    g_nstage = 0;
}

static void tex_fill(vtex *v, const tex_t *t);   // gfx_vk_draw.inc: a texture the GPU has not written gets its guest-memory content

// ---------------------------------------------------------------- operations
static void clear_color(const p_clear_color *c) {
    if (!R.has_rt || c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) return;
    vtex *v = use(R.rt.color[c->index]); if (!v) return;
    v->gpu = 1;
    static int warned; if (c->mask != 0xf && !warned++) LOG("[vk] a colour clear with channel mask %#x was done on all channels (stage B has no masked clears)", c->mask);
    uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.cview[c->index], R.rt.has_cview[c->index], &lv, &nl, &l0, &nlay);
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, l0, nlay};
    VkClearColorValue cv; memcpy(cv.float32, c->rgba, 16);
    if (fmt_int(v->fmt)) for (int i = 0; i < 4; i++) cv.uint32[i] = (uint32_t)c->rgba[i];
    vkCmdClearColorImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rr); full_barrier();
}
static void clear_ds(const p_clear_ds *c) {
    if (!R.has_rt || !R.rt.depth) return;
    vtex *v = use(R.rt.depth); if (!v) return;
    v->gpu = 1;
    VkImageAspectFlags a = fmt_aspects(v->fmt) & ((c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
    if (!a) return;
    uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.dview, R.rt.has_dview, &lv, &nl, &l0, &nlay);
    VkImageSubresourceRange rr = {a, lv, 1, l0, nlay};
    VkClearDepthStencilValue z = {c->depth, (uint32_t)c->stencil};
    vkCmdClearDepthStencilImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); full_barrier();
}
static void clear_tex(const p_clear_tex *c) {
    vtex *v = use(c->tex); if (!v) return;
    v->gpu = 1;
    uint32_t lv, nl, l0, nlay; view_range(v, &c->view, c->has_view, &lv, &nl, &l0, &nlay);
    uint32_t w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1;
    static int warned; if ((c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width < w || (uint32_t)c->r.height < h) && !warned++) LOG("[vk] a texture clear of part of a level was done on the whole level (stage B)");
    VkImageSubresourceRange rr = {fmt_aspects(v->fmt), lv, 1, l0 + (v->is3d ? 0 : (uint32_t)c->r.zoffset), v->is3d ? 1 : (c->r.depth > 0 ? (uint32_t)c->r.depth : nlay)};
    if (rr.baseArrayLayer + rr.layerCount > v->layers) rr.layerCount = v->layers - rr.baseArrayLayer;
    if (fmt_depth(v->fmt)) { VkClearDepthStencilValue z = {c->c.f[0], c->c.u[1]}; vkCmdClearDepthStencilImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    else { VkClearColorValue cv; memcpy(&cv, &c->c, 16); vkCmdClearColorImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rr); }
    full_barrier();
}
static void copy_b2t(const p_copy_b2t *c) {
    vtex *v = use(c->tex); if (!v) return;
    tex_fill(v, c->tex); v->gpu = 1;   // a copy into part of a texture keeps the rest: its guest content goes in first
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    uint32_t lv, nl, l0, nlay; view_range(v, &c->view, c->has_view, &lv, &nl, &l0, &nlay);
    size_t wb = (size_t)((c->r.width + bw - 1) / bw) * bpb, rows = (size_t)(c->r.height + bh - 1) / bh;
    size_t row = R.copy_stride ? (size_t)R.copy_stride : wb, img = R.copy_img_stride ? (size_t)R.copy_img_stride : row * rows;
    int nz = c->r.depth > 0 ? c->r.depth : 1;
    // Vulkan reads rows of a whole number of texel blocks: the rows are packed tightly into the staging buffer, which also takes them out of the guest's memory now (in record order)
    stage_t *s = stage_new(wb * rows * nz); if (!s) return;
    const uint8_t *src = (const uint8_t *)(uintptr_t)c->src;
    for (int z = 0; z < nz; z++) for (size_t r = 0; r < rows; r++) memcpy((uint8_t *)s->p + (z * rows + r) * wb, src + z * img + r * row, wb);
    VkBufferImageCopy bic = {0, 0, 0, {fmt_aspects(v->fmt) & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT), lv, v->is3d ? 0 : l0 + (uint32_t)c->r.zoffset, v->is3d ? 1 : (uint32_t)nz},
                             {c->r.xoffset, c->r.yoffset, v->is3d ? c->r.zoffset : 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, v->is3d ? (uint32_t)nz : 1}};
    if (!bic.imageSubresource.aspectMask) bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    if (fmt_aspects(v->fmt) == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { static int w; if (!w++) LOG("[vk] a buffer copy into a depth-stencil texture took its depth plane only (stage B)"); }
    vkCmdCopyBufferToImage_(g_cb, s->b, v->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bic); full_barrier();
}
static void copy_t2b(const p_copy_t2t *c) {   // T2B reuses p_copy_t2t: src holds the destination buffer address, dst the texture
    const tex_t *t = c->dst; uint64_t dst = (uint64_t)(uintptr_t)c->src;
    vtex *v = use(t); if (!v) return;
    tex_fill(v, t);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) return;
    uint32_t lv, nl, l0, nlay; view_range(v, &c->dv, c->has_dv, &lv, &nl, &l0, &nlay);
    const NVNcopyRegion *r = &c->dr;
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, rows = (size_t)(r->height + bh - 1) / bh;
    size_t row = R.copy_stride ? (size_t)R.copy_stride : wb; int nz = r->depth > 0 ? r->depth : 1;
    if (nz > 1) { static int w; if (!w++) LOG("[vk] a texture → buffer copy of %d layers copied the first only (stage B)", nz); }
    stage_t *s = stage_new(wb * rows); if (!s) return;
    s->rb_dst = (uint8_t *)(uintptr_t)dst; s->rb_row = row; s->rb_src_row = wb; s->rb_rows = rows; s->rb_n = wb;
    VkBufferImageCopy bic = {0, 0, 0, {fmt_aspects(v->fmt) & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT), lv, v->is3d ? 0 : l0 + (uint32_t)r->zoffset, 1},
                             {r->xoffset, r->yoffset, v->is3d ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, 1}};
    if (!bic.imageSubresource.aspectMask) bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    vkCmdCopyImageToBuffer_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, s->b, 1, &bic); full_barrier();
}
static void copy_t2t(const p_copy_t2t *c) {
    vtex *s = use(c->src), *d = use(c->dst); if (!s || !d) return;
    tex_fill(s, c->src); tex_fill(d, c->dst); d->gpu = 1;
    uint32_t slv, snl, sl0, snlay, dlv, dnl, dl0, dnlay;
    view_range(s, &c->sv, c->has_sv, &slv, &snl, &sl0, &snlay); view_range(d, &c->dv, c->has_dv, &dlv, &dnl, &dl0, &dnlay);
    const NVNcopyRegion *a = &c->sr, *b = &c->dr;
    int nz = a->depth > 0 ? a->depth : 1;
    VkImageAspectFlags sa = fmt_aspects(s->fmt), da = fmt_aspects(d->fmt);
    int same = a->width == b->width && a->height == b->height && !(c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT));
    int compressed = s->fmt >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && s->fmt <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    if (same && (s->fmt == d->fmt || compressed)) {
        VkImageCopy ic = {{sa, slv, s->is3d ? 0 : sl0 + (uint32_t)a->zoffset, s->is3d ? 1 : (uint32_t)nz}, {a->xoffset, a->yoffset, s->is3d ? a->zoffset : 0},
                          {da, dlv, d->is3d ? 0 : dl0 + (uint32_t)b->zoffset, d->is3d ? 1 : (uint32_t)nz}, {b->xoffset, b->yoffset, d->is3d ? b->zoffset : 0},
                          {(uint32_t)a->width, (uint32_t)a->height, s->is3d ? (uint32_t)nz : 1}};
        vkCmdCopyImage_(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    } else {
        // A scaled, mirrored or format-changing copy: a blit (filtered when the game asks LINEAR_FILTER)
        int x0 = a->xoffset, x1 = a->xoffset + a->width, y0 = a->yoffset, y1 = a->yoffset + a->height;
        if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { int t = x0; x0 = x1; x1 = t; }
        if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { int t = y0; y0 = y1; y1 = t; }
        VkImageBlit bl = {{sa, slv, s->is3d ? 0 : sl0 + (uint32_t)a->zoffset, s->is3d ? 1 : (uint32_t)nz}, {{x0, y0, 0}, {x1, y1, 1}},
                          {da, dlv, d->is3d ? 0 : dl0 + (uint32_t)b->zoffset, d->is3d ? 1 : (uint32_t)nz}, {{b->xoffset, b->yoffset, 0}, {b->xoffset + b->width, b->yoffset + b->height, 1}}};
        vkCmdBlitImage_(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && !fmt_depth(s->fmt) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    full_barrier();
}

// ---------------------------------------------------------------- stage C: draws and compute
static uint64_t g_draws, g_dispatches, g_draws_ready, g_disp_ready;
static const vks_prog *g_gfx_prog, *g_cs_prog;   // the bound programs' shaders (stage C), NULL while a program's shaders are not known
static int g_draw_on = 1;   // DM_VK_DRAW=0: count draws without drawing (stage B's behaviour)
#include "gfx_vk_draw.inc"

// ---------------------------------------------------------------- the record walk
void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); mtl_exec((const uint8_t *)(uintptr_t)t); break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); post((post_op){1, val, m, v, 0, 0}); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); post((post_op){2, NULL, 0, 0, ad, (uint32_t)t}); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = atomic_load(&g_fence) + 1; s->fenced = 1; break; }
        case OP_CLEAR_BUF: { const p_clear_buf *c = (const p_clear_buf *)a; uint32_t *d = (uint32_t *)(uintptr_t)c->addr; for (uint64_t i = 0; i < c->size / 4; i++) d[i] = c->value; break; }
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); memmove((void *)(uintptr_t)d, (const void *)(uintptr_t)s_, n); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); memcpy((void *)(uintptr_t)d, a + 16, n); break; }
        case OP_COPY_STRIDE: { const p_copy_stride *cs = (const p_copy_stride *)a; if (cs->image) R.copy_img_stride = cs->stride; else R.copy_stride = cs->stride; break; }
        case OP_SET_RT: if (VK.ok) pass_end(); R.rt = *(const p_set_rt *)a; R.has_rt = 1; break;
        case OP_CLEAR_COLOR: if (VK.ok) { pass_end(); clear_color((const p_clear_color *)a); } break;
        case OP_CLEAR_DS: if (VK.ok) { pass_end(); clear_ds((const p_clear_ds *)a); } break;
        case OP_CLEAR_TEX: if (VK.ok) { pass_end(); clear_tex((const p_clear_tex *)a); } break;
        case OP_COPY_B2T: if (VK.ok) { pass_end(); copy_b2t((const p_copy_b2t *)a); } break;
        case OP_COPY_T2B: if (VK.ok) { pass_end(); copy_t2b((const p_copy_t2t *)a); } break;
        case OP_COPY_T2T: if (VK.ok) { pass_end(); copy_t2t((const p_copy_t2t *)a); } break;
        case OP_BIND_PROGRAM: if (VK.ok) {
            const p_bind_program *b = (const p_bind_program *)a; const vks_prog *pr = vks_prog_get(b->prog);
            if (pr && pr->compute) g_cs_prog = pr; else if (pr || !(b->stages & NVN_SHADER_STAGE_COMPUTE_BIT)) g_gfx_prog = pr; else g_cs_prog = NULL;
        } break;
        case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: g_draws++; if (g_gfx_prog) g_draws_ready++; if (VK.ok && g_draw_on) draw_op(h.op, a, h.size); break;
        case OP_DRAW_TEX: g_draws++; break;
        case OP_DISPATCH: g_dispatches++; if (g_cs_prog) g_disp_ready++; if (VK.ok && g_draw_on) dispatch_op((const p_dispatch *)a); break;
        case OP_BARRIER: break;   // every pass end and GPU operation is followed by a full barrier already
        default: state_op(h.op, a); break;   // state for draws and compute
        }
        p += h.size;
    }
}
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }

void mtl_sub_begin(void) {
    // the window comes up with the first submission, so the game's textures have a device from the start
    if (vkw_wanted() && !VK.ok) { vkw_init(); const char *e = getenv("DM_VK_DRAW"); g_draw_on = !(e && *e == '0'); if (VK.ok) LOG("[vk] draws: %s", g_draw_on ? "on (stage C)" : "off (DM_VK_DRAW=0)"); }
}
uint64_t mtl_sub_commit(void) {
    if (g_cb_open) {
        pass_end();
        vkEndCommandBuffer_(g_cb); g_cb_open = 0;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &g_cb, 0, NULL};
        pthread_mutex_lock(&VK.qmu);
        VkResult r = vkQueueSubmit_(VK.q, 1, &si, g_fence_vk);
        pthread_mutex_unlock(&VK.qmu);
        if (r) LOG("[RED] [vk] submission failed (%d)", r);
        else {
            VkResult w = vkWaitForFences_(VK.dev, 1, &g_fence_vk, VK_TRUE, 5000000000ull);
            if (w) { LOG("[RED] [vk] the GPU did not finish a submission in 5 s (%d): device lost or hung", w); vkDeviceWaitIdle_(VK.dev); }
            vkResetFences_(VK.dev, 1, &g_fence_vk);
        }
        stages_done();
        draw_submitted();
    }
    // The submission has completed: its events and counters land, in record order
    uint64_t now = gpu_ticks();
    for (int i = 0; i < g_npost; i++) {
        post_op *o = &g_post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else {
            // samples passed (occlusion queries): no draws yet, so every query is reported visible rather than culling everything; timestamps: the guest clock in GPU ticks
            uint64_t rec[2] = {o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? now : 0, now};
            memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
        }
    }
    g_npost = 0;
    static uint64_t reported;
    uint64_t f = atomic_fetch_add(&g_fence, 1) + 1;
    if (f == 1 || f - reported >= 600) {
        reported = f; uint64_t np = 0, nm = 0, nf = 0; vks_stats(&np, &nm, &nf);
        char ds[640]; draw_summary(ds, sizeof ds);
        LOG("[vk] replay: %llu submissions; draws %llu (%llu with their shaders ready), dispatches %llu (%llu ready); programs %llu, shaders missing %llu, refused %llu; %s",
            (unsigned long long)f, (unsigned long long)g_draws, (unsigned long long)g_draws_ready, (unsigned long long)g_dispatches, (unsigned long long)g_disp_ready,
            (unsigned long long)np, (unsigned long long)nm, (unsigned long long)nf, ds);
    }
    return f;
}
uint64_t mtl_last_fence(void) { return atomic_load(&g_fence); }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { (void)timeout_ns; return fence <= atomic_load(&g_fence); }   // every submission completes before mtl_sub_commit returns
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
void mtl_drain(void) {}
void mtl_finish(void) {}

// The image to show for a present (gfx_null.c mtl_present): the swapchain texture's VkImage, or none (stage A's colour) while nothing has written it
int vk_present_image(const tex_t *t, VkImage *img, uint32_t *w, uint32_t *h) {
    if (!t || !VK.ok) return 0;
    pthread_mutex_lock(&g_tex_mu); vtex *v = tex_slot(t->serial, 0); int ok = v && v->img && (v->bad & 2);
    if (ok) { *img = v->img; *w = v->w; *h = v->h; }
    pthread_mutex_unlock(&g_tex_mu);
    return ok;
}
