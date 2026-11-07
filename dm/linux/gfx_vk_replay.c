// Stage B of the Vulkan graphics layer (dm/linux/GRAPHICS.md): the replay of the game's recorded NVN commands (nvn_obj.h records, as dm/gfx/nvn/nvn_mtl.m replays them on Metal).
//
// What it does now:
//   - the CPU-visible effects, with or without a window: events (SignalEvent), counters (ReportCounter), fence syncs, buffer copies, fills and inline uniform updates. Buffers are the
//     guest's own memory, which the CPU reads and writes directly, so these run on the CPU in record order; the effects the game waits for (events, counters) land when the submission
//     completes, as on the GPU
//   - with a window (DM_WINDOW=1): textures as VkImages (by serial, made at first use), colour and depth / stencil clears of the render targets, texture clears, buffer → texture,
//     texture → texture and texture → buffer copies, and the present of the game's swapchain image
// What it does not do yet: draws and compute (stage C: shaders), packaged texture uploads (their storage is block-linear), partial clears (a scissor, a channel mask), sparse textures.
// Every image stays in VK_IMAGE_LAYOUT_GENERAL, and every GPU operation is followed by a full memory barrier: simple and correct, not fast. A submission runs on the GPU while the next one is recorded (two slots: gfx_vk_draw.inc and the end of this file); DM_VK_ASYNC=0 waits for each.
#include "dm.h"
#include "../gfx/nvn/nvn_obj.h"
#include "../gfx/nvn/nvn_mtl.h"
#include "../gfx/nvn/nvn_timestamp.h"
#include "../gfx/tex/nvn_fmt_mtl.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include "../gfx/nvn/nvn_dirty.h"
#include "gfx_vk.h"
#include <stdatomic.h>
#include <math.h>
#include <time.h>

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

// ---------------------------------------------------------------- textures: one VkImage per texture serial (made at first use: a render target's first use decides whether it is scaled)
struct vview;
// gpu: the GPU has written it (render target, clear, copy destination, shader image): its content lives here and is never replaced from guest memory;
// uploaded: its guest-memory content was copied in (stage C uploads a texture the GPU has not written at its first sampling, and again after the CPU writes it: mtl_tex_cpu_written)
// linear: a LINEAR texture, which the CPU and GPU may both write: its guest memory stays the reference (GPU writes are read back into it after each submission: wrote)
// tracked: its guest pages are write-protected since stamp (nvn_dirty.c), so a CPU write since its upload is seen (prep_tex) and it is uploaded again
typedef struct { uint64_t serial; VkImage img; VkDeviceMemory mem; VkFormat fmt; uint32_t w, h, d, layers, levels, samples; int is3d, is1d, bad, gpu, uploaded, storage; struct vview *views;
                 int linear, wrote, tracked, mutable_fmt, cube, twin; uint64_t alias; uint32_t gw, gh; float sc; const void *gkey; uint8_t *gmem; uint64_t gsize, stamp; const void *span; uint32_t stride, rowbytes, rows; uint64_t bytes; } vtex;
static vtex **g_lw; static int g_nlw, g_clw;   // linear textures the GPU wrote in this submission
static inline void gpu_wrote(vtex *v) {
    v->gpu = 1; v->uploaded = 1;   // the GPU's content is the newest now: nothing from guest memory goes over it (a linear texture's is read back at the submission's end)
    if (v->linear && !v->wrote) { v->wrote = 1; if (g_nlw == g_clw) { g_clw = g_clw ? 2 * g_clw : 16; g_lw = realloc(g_lw, g_clw * sizeof *g_lw); } g_lw[g_nlw++] = v; }
}
#define TEX_CAP (1u << 16)
static vtex g_tex[TEX_CAP];   // open addressing by serial
static pthread_mutex_t g_tex_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_tex_made, g_tex_bytes, g_tex_live, g_tex_live_bytes;
static vtex *tex_slot(uint64_t serial, int make) {
    uint32_t i = (uint32_t)(serial * 0x9E3779B97F4A7C15ull >> 48) & (TEX_CAP - 1);
    for (uint32_t k = 0; k < TEX_CAP; k++, i = (i + 1) & (TEX_CAP - 1)) {
        if (g_tex[i].serial == serial) return &g_tex[i];
        if (!g_tex[i].serial || g_tex[i].serial == ~0ull) { if (!make) return NULL; if (!g_tex[i].serial) { memset(&g_tex[i], 0, sizeof g_tex[i]); g_tex[i].serial = serial; return &g_tex[i]; } }
    }
    return NULL;
}
// The sRGB / UNORM twin of a format (0: none): two textures over the same memory in these two formats are one image (a game draws into an sRGB texture and
// reads it back through a UNORM texture of the same storage, as Mario Tennis Fever's upscaling does)
static VkFormat fmt_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32; case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// Where a texture's storage is, for finding textures over the same storage: its memory, or for a texture made before its (virtual) pool had memory, the pool's
// memory now, or failing that the pool and offset themselves
static const void *tex_key(const tex_t *t) {
    if (t->mem) return t->mem;
    if (t->pool && t->pool->mem) return t->pool->mem + t->off;
    if (t->pool) return (const uint8_t *)t->pool + ((uint64_t)t->off << 1 | 1);   // odd: never a real address of a texture's storage
    return NULL;
}
static VkFormat tex_vkfmt(const tex_t *t) { uint32_t mf = 0; uint8_t swz[4]; nvn_fmt_to_mtl(t->d.format, &mf, swz); return vk_of_mtl(mf); }
// The format a texture's views take of its image: its own, when the image is another texture's (an alias) in the twin format
static VkFormat nvn_vkfmt(uint32_t nf) { uint32_t mf = 0; uint8_t swz[4]; nvn_fmt_to_mtl(nf, &mf, swz); return vk_of_mtl(mf); }
// ... or the format a texture view asks for (nvnTextureViewSetFormat: a game draws into an sRGB texture through a UNORM view so its shader's sRGB values are
// stored as they are, as Mario Tennis Fever's upscale into the window's texture does)
static VkFormat tex_view_fmt2(const vtex *v, const tex_t *t, const view_t *vw, int has_vw) {
    if (!v->twin) return v->fmt;
    VkFormat f = has_vw && vw && vw->set_format ? nvn_vkfmt(vw->format) : t ? tex_vkfmt(t) : VK_FORMAT_UNDEFINED;
    return f && f == fmt_twin(v->fmt) ? f : v->fmt;
}
static VkFormat tex_view_fmt(const vtex *v, const tex_t *t) { return tex_view_fmt2(v, t, NULL, 0); }
// Guest storage → the textures whose images hold it (aliases find their image here); entries are checked against the texture when read. Several live textures may start
// at the same storage (a game's transient render target memory: a 960x540 R11G11B10 target and the 1920x1080 RGBA8 picture at one address), so each keeps its own
// entry: with one entry per address, the newest texture there hid the others, and an sRGB view of the picture made after it got an image of its own, never drawn:
// Mario Tennis Fever's picture went black (now and then: it depends on the order the game makes its textures)
#define GM_CAP (1u << 15)
static struct { const void *m; uint64_t serial; } g_gm[GM_CAP];
static uint32_t gm_hash(const void *m) { return (uint32_t)((uint64_t)(uintptr_t)m * 0x9E3779B97F4A7C15ull >> 49) & (GM_CAP - 1); }
static int gm_dead(uint32_t i) { vtex *b = tex_slot(g_gm[i].serial, 0); return !b || !b->img || b->gkey != g_gm[i].m; }
static void gm_put(const void *m, uint64_t serial) {
    uint32_t i = gm_hash(m), free_i = ~0u;
    for (uint32_t k = 0; k < 128; k++, i = (i + 1) & (GM_CAP - 1)) {
        if (g_gm[i].m == m && g_gm[i].serial == serial) return;
        if (free_i == ~0u && (!g_gm[i].m || gm_dead(i))) free_i = i;
        if (!g_gm[i].m) break;
    }
    if (free_i != ~0u) { g_gm[free_i].m = m; g_gm[free_i].serial = serial; }
    else { static int n; if (n++ < 5) LOG("[YELLOW] [vk] texture storage map: no room near %p (serial %llu not recorded for aliasing)", m, (unsigned long long)serial); }
}
// A live texture at storage m whose image another of this shape can share (w ~0: any): the most recently made one (highest serial)
static vtex *gm_find(const void *m, uint32_t w, uint32_t h, uint32_t layers, uint32_t levels, uint32_t samples, int is3d, VkFormat f, vtex **any) {
    uint32_t i = gm_hash(m); vtex *best = NULL; *any = NULL;
    for (uint32_t k = 0; k < 128 && g_gm[i].m; k++, i = (i + 1) & (GM_CAP - 1)) {
        if (g_gm[i].m != m || gm_dead(i)) continue;
        vtex *b = tex_slot(g_gm[i].serial, 0);
        if (!*any || b->serial > (*any)->serial) *any = b;
        int ok = b->gw == w && b->gh == h && b->layers == layers && b->levels >= levels && b->samples == samples && b->is3d == is3d && (f == b->fmt || (b->twin && f == fmt_twin(b->fmt)));
        if (ok && !b->alias && (!best || b->serial > best->serial)) best = b;
    }
    return best;
}
// The texture's vtex, following an alias to the image it shares
static vtex *tex_find(uint64_t serial) {
    vtex *v = tex_slot(serial, 0);
    if (v && v->alias) { vtex *b = tex_slot(v->alias, 0); if (b && b->img) return b; }
    return v;
}
static _Atomic int g_dsc_stale;   // gfx_vk_draw.inc
static int g_rt_hint;   // the texture asked for is about to be a render target (a pass's attachment, a target clear): only those may be scaled (resolution scaling)
static vtex *tex_get(const tex_t *t) {
    if (!t || !t->serial || !VK.ok) return NULL;
    pthread_mutex_lock(&g_tex_mu);
    vtex *v = tex_slot(t->serial, 1);
    if (v && v->alias) {   // another texture's image: while that texture lives with the same storage
        vtex *b = tex_slot(v->alias, 0);
        if (b && b->img && b->gkey == tex_key(t)) { pthread_mutex_unlock(&g_tex_mu); return b; }
        v->alias = 0;
    }
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
    // shader images (nvnCommandBufferBindImage): only textures made for it (NVN_TEXTURE_FLAGS_IMAGE, which NVN asks of a texture bound as an image). Storage use costs
    // Adreno its framebuffer compression (UBWC) on most formats, and every render target paid for it. DM_VK_STORAGE_ALL=1: every texture whose format allows it, as before
    static int sall = -1; if (sall < 0) { const char *e = getenv("DM_VK_STORAGE_ALL"); sall = e && *e == '1'; }
    if ((ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && samples == 1 && (sall || (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE))) { ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT; v->storage = 1; }
    // A texture registered as an image: a shader may read or write its texels as the other numeric kind (an R32_UINT target written with floats, no answer table to say
    // otherwise): its views may take another format of the same size (gfx_vk_draw.inc tex_view), and a usage only that format has
    if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !depth && !(f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)) {
        ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; v->mutable_fmt = 1;
        if (samples == 1 && !v->storage) { ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT; v->storage = 2; }   // storage through a view of another format (an sRGB target written as UNORM)
    }
    if (ci.imageType == VK_IMAGE_TYPE_2D && ci.arrayLayers >= 6 && ci.extent.width == ci.extent.height && samples == 1) { ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; v->cube = 1; }
    if (samples > 1) ci.mipLevels = 1;
    // Another texture over the same storage, of the same shape, in this format or its twin: this texture is that one's image (an alias)
    const void *key = tex_key(t);
    if (key && !depth) {
        vtex *other = NULL;
        vtex *b = gm_find(key, ci.extent.width, ci.extent.height, ci.arrayLayers, ci.mipLevels, samples, ci.imageType == VK_IMAGE_TYPE_3D, f, &other);
        if (b == v) b = NULL;
        if (!b) b = other != v ? other : NULL;
        if (b) {
            int shape = b->gw == ci.extent.width && b->gh == ci.extent.height && b->layers == ci.arrayLayers && b->levels >= ci.mipLevels && b->samples == samples && b->is3d == (ci.imageType == VK_IMAGE_TYPE_3D);
            if (shape && (f == b->fmt || (b->twin && f == fmt_twin(b->fmt)))) {
                v->alias = b->serial;
                static int n; if (n++ < 200) LOG("[vk] texture serial %llu (format %d) shares the image of serial %llu (format %d): the same storage", (unsigned long long)t->serial, f, (unsigned long long)b->serial, b->fmt);
                pthread_mutex_unlock(&g_tex_mu); return b;
            }
            static int n2; if (n2++ < 30) LOG("[YELLOW] [vk] texture serial %llu (format %d %ux%u x%u levels %u) is over the storage of serial %llu (format %d %ux%u x%u levels %u) but cannot share its image: it gets its own",
                                              (unsigned long long)t->serial, f, ci.extent.width, ci.extent.height, ci.arrayLayers, ci.mipLevels, (unsigned long long)b->serial, b->fmt, b->w, b->h, b->layers, b->levels);
        }
    }
    // Resolution scaling (DM_VK_SCALE, e.g. 0.75; DM_VK_SCALE_SIZE, default 1920x1080): the game's full-size render targets get images that much smaller, and everything
    // drawn into them and read from them follows (the pass, its viewports and scissors, copies; the translated shaders take the scales from the support buffer, as
    // Ryujinx's resolution scaler: fragment coordinates, texel fetches and texture sizes). Only single-level 2D textures of exactly that size, not linear, not shader images
    uint32_t gw = ci.extent.width, gh = ci.extent.height; float sc = 1.0f;
    {
        static float g_sc = -1; static uint32_t sw = 1920, sh = 1080;
        if (g_sc < 0) { const char *e = getenv("DM_VK_SCALE"); g_sc = e ? (float)atof(e) : 1.0f; if (!(g_sc > 0.2f && g_sc < 1.0f)) g_sc = 1.0f;
            const char *z = getenv("DM_VK_SCALE_SIZE"); if (z) sscanf(z, "%ux%u", &sw, &sh);
            if (g_sc != 1.0f) LOG("[vk] resolution scale %.2f for %ux%u render targets (DM_VK_SCALE, DM_VK_SCALE_SIZE)", g_sc, sw, sh); }
        // a render target: its first use is as one, or NVN made it compressible (render targets are; textures loaded from files are not)
        if (g_sc != 1.0f && gw == sw && gh == sh) {
            int rt = g_rt_hint || (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE);
            const char *why = !rt ? "not a render target" : ci.imageType != VK_IMAGE_TYPE_2D || ci.arrayLayers != 1 ? "not a single 2D image" : ci.mipLevels != 1 ? "mipmapped" : samples != 1 ? "multisampled" :
                              (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) ? "linear" : NULL;
            if (!why) { sc = g_sc; ci.extent.width = (uint32_t)(gw * sc + 0.5f); ci.extent.height = (uint32_t)(gh * sc + 0.5f); }
            static int n4; if (n4++ < 60) LOG("[vk] texture serial %llu (format %d %ux%u, flags %#x, first use as a target %d): %s%s", (unsigned long long)t->serial, f, gw, gh, t->d.flags, g_rt_hint,
                                              why ? "not scaled: " : "scaled", why ? why : "");
        }
    }
    if (ci.extent.width >= 1280) { static int n3; if (n3++ < 80) LOG("[vk] large texture serial %llu: format %d %ux%u x%u levels %u, storage %p (memory %p, pool %p + %#llx)", (unsigned long long)t->serial, f, ci.extent.width, ci.extent.height, ci.arrayLayers, ci.mipLevels, key, (void *)t->mem, (void *)t->pool, (unsigned long long)t->off); }
    // Formats with an sRGB / UNORM twin: the image can take views of both (the format list keeps the GPU's compression)
    VkFormat twin = fmt_twin(f); VkFormat fl[2] = {f, twin};
    VkImageFormatListCreateInfo fli = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, fl};
    if (twin && samples >= 1) {
        if (!v->mutable_fmt) {
            ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; fli.pNext = ci.pNext; ci.pNext = &fli;
            if (ci.usage & VK_IMAGE_USAGE_STORAGE_BIT) ci.flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;   // the sRGB twin cannot be a shader image: only views of this format are
        }
        v->twin = 1;
    }
    if (!ff) { v->bad = 1; pthread_mutex_unlock(&g_tex_mu); static int once2; if (!once2++) LOG("[RED] [vk] the GPU has no support for Vulkan format %d (NVN %#x): such textures are skipped (BC formats on Adreno need a CPU decode)", f, t->d.format); return NULL; }
    VkResult r = vkCreateImage_(VK.dev, &ci, NULL, &v->img);
    if (r) { v->bad = 1; v->img = VK_NULL_HANDLE; pthread_mutex_unlock(&g_tex_mu); LOG("[RED] [vk] vkCreateImage %ux%ux%u x%u levels %u format %d failed (%d)", ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, f, r); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements_(VK.dev, v->img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    if (ai.memoryTypeIndex == ~0u) ai.memoryTypeIndex = vk_mem_type(mr.memoryTypeBits, 0);
    if ((r = vkAllocateMemory_(VK.dev, &ai, NULL, &v->mem)) || (r = vkBindImageMemory_(VK.dev, v->img, v->mem, 0))) {
        LOG("[RED] [vk] texture memory (%.1f MB) failed (%d)", mr.size / 1048576.0, r); vkDestroyImage_(VK.dev, v->img, NULL); v->img = VK_NULL_HANDLE; v->bad = 1; pthread_mutex_unlock(&g_tex_mu); return NULL; }
    // where its content lives in guest memory (uploads, CPU-write tracking, a linear texture's read-back)
    if (t->mem) {
        nvn_texlayout L = t->L; if (!L.size) nvn_tex_layout(&t->d, &L);
        int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        v->gmem = t->mem; v->gsize = L.size;
        v->linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && ci.mipLevels == 1 && ci.arrayLayers == 1 && ci.imageType != VK_IMAGE_TYPE_3D && bpb && samples == 1;
        if (v->linear) { v->rowbytes = (ci.extent.width + bw - 1) / bw * (uint32_t)bpb; v->rows = (ci.extent.height + bh - 1) / bh; v->stride = t->d.stride ? t->d.stride : v->rowbytes; if (!v->gsize) v->gsize = (uint64_t)v->stride * v->rows; }
    }
    v->gkey = key;
    if (key && !depth) gm_put(key, t->serial);
    v->gw = gw; v->gh = gh; v->sc = sc;
    v->fmt = f; v->w = ci.extent.width; v->h = ci.extent.height; v->d = ci.extent.depth; v->layers = ci.arrayLayers; v->levels = ci.mipLevels; v->is3d = ci.imageType == VK_IMAGE_TYPE_3D; v->is1d = ci.imageType == VK_IMAGE_TYPE_1D; v->samples = samples;
    g_tex_made++; g_tex_bytes += mr.size; g_tex_live++; g_tex_live_bytes += mr.size; v->bytes = mr.size;
    if (g_tex_made == 1 || g_tex_made % 500 == 0) LOG("[vk] textures made: %llu, %.0f MB", (unsigned long long)g_tex_made, g_tex_bytes / 1048576.0);
    pthread_mutex_unlock(&g_tex_mu);
    return v;
}
static void vk_views_drop(vtex *v);
void mtl_tex_init(const tex_t *t) { (void)t; }   // made at first use (tex_get): most textures a game initializes are packaged textures stage B does not upload yet
static void sub_wait_all(void);
void mtl_tex_release(tex_t *t) {
    if (!t || !t->serial) return;
    sub_wait_all();   // a submission in flight may still read its storage back into guest memory: done first, so nothing lands in memory the game reuses
    atomic_store(&g_dsc_stale, 1);   // descriptor sets that may hold its views are not reused (gfx_vk_draw.inc ws_commit)
    pthread_mutex_lock(&g_tex_mu);
    vtex *v = tex_slot(t->serial, 0);
    if (v) {
        if (v->alias) { memset(v, 0, sizeof *v); v->serial = ~0ull; pthread_mutex_unlock(&g_tex_mu); return; }   // owns no image
        if (v->img) { pthread_mutex_lock(&VK.qmu); vkQueueWaitIdle_(VK.q); pthread_mutex_unlock(&VK.qmu); vk_views_drop(v); vkDestroyImage_(VK.dev, v->img, NULL); vkFreeMemory_(VK.dev, v->mem, NULL); g_tex_live--; g_tex_live_bytes -= v->bytes; }
        memset(v, 0, sizeof *v); v->serial = ~0ull;   // a tombstone keeps later probes going
    }
    pthread_mutex_unlock(&g_tex_mu);
}

// ---------------------------------------------------------------- the submission being recorded: its command buffer and fence (the one in flight has the other pair: subslot, below)
static VkCommandPool g_pool; static VkCommandBuffer g_cb, g_pre; static VkFence g_fence_vk; static int g_cb_open, g_cb_ops, g_pre_open;   // g_pre: GPU copies that go before g_cb (forwarding)
static _Atomic uint64_t g_fence;   // submissions handed to the GPU (mtl_last_fence)
static _Atomic uint64_t g_done;    // submissions completed: their read-backs, write-backs, events and counters are in guest memory
// Guest bytes each submission in flight writes when it completes (read-backs, storage buffer write-backs, events and counters). The replay reads and writes guest memory
// on the CPU as it records (ring copies, uploads, indirect arguments, buffer copies and updates): where that touches these bytes, that submission is waited for first,
// so the CPU work sees (and comes after) what the GPU wrote, in submission order. Built and read by the recording thread only (busy: set by it, cleared at completion)
#define NSUB 3   // submission slots: the one being recorded and up to two in flight
typedef struct { uint64_t lo, hi; } grange;
static struct { grange *r; int n, cap; uint64_t lo, hi; _Atomic int busy; } g_pr[NSUB - 1];
static uint64_t g_hazard_waits;
static void pend_wait(void); static void pend_wait_slot(int i);
static struct { const char *file; int line; uint64_t n, bytes, ns; } g_hz[16];   // the waits by the replay line that read the bytes (logged with the time per submission)
static uint64_t mono_ns(void);
static void hz_note(const char *file, int line, uint64_t n, uint64_t ns) {
    for (int i = 0; i < 16; i++) if ((g_hz[i].line == line && g_hz[i].file == file) || !g_hz[i].line) { g_hz[i].file = file; g_hz[i].line = line; g_hz[i].n++; g_hz[i].bytes += n; g_hz[i].ns += ns; return; }
}
static inline void pend_hazard_(uint64_t addr, uint64_t n, const char *file, int line) {
    if (!addr || !n) return;
    for (int s = 0; s < NSUB - 1; s++) {
        if (!atomic_load_explicit(&g_pr[s].busy, memory_order_acquire) || addr >= g_pr[s].hi || addr + n <= g_pr[s].lo) continue;
        for (int i = 0; i < g_pr[s].n; i++) if (addr < g_pr[s].r[i].hi && g_pr[s].r[i].lo < addr + n) { g_hazard_waits++; uint64_t t0 = mono_ns(); pend_wait_slot(s); hz_note(file, line, n, mono_ns() - t0); break; }
    }
}
#define pend_hazard(a, n) pend_hazard_((a), (n), __FILE__, __LINE__)
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // 1 event, 2 counter
static post_op *g_post; static int g_npost, g_postcap;
static struct { p_set_rt rt; int has_rt; int64_t copy_stride, copy_img_stride; } R;
// Staging buffers of this submission (texture uploads and readbacks), freed at commit; readbacks copy into guest memory after the GPU finishes
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint64_t n; uint8_t *rb_dst; size_t rb_row, rb_src_row, rb_rows, rb_n; int dump_fmt, lin_rb; uint32_t dw, dh; char dump_name[160]; uint8_t *orig; int64_t fwo; uint32_t fw_epoch; } stage_t;   // orig, fwo: forwarding (gfx_vk_draw.inc)
static void dump_write(const stage_t *s);   // gfx_vk_draw.inc: a frame dump's image (DM debug: L3 + R3 or F12)
static stage_t *g_stage; static int g_nstage, g_stagecap;

static void post(post_op o) { if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); } g_post[g_npost++] = o; }
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; } return nvn_timestamp_from_ns(g); }

static void prof_cb_begin(void); static void prof_cb_end(void); static void prof_report(void);   // gfx_vk_draw.inc: the GPU profile
static int cb_begin(void) {
    if (!VK.ok) return 0;
    if (!g_pool) {
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VK.qfam};
        vkCreateCommandPool_(VK.dev, &pci, NULL, &g_pool);
    }
    if (!g_cb) {   // one per slot: the second is made when the first is in flight
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkAllocateCommandBuffers_(VK.dev, &cai, &g_cb);
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0}; vkCreateFence_(VK.dev, &fi, NULL, &g_fence_vk);
    }
    if (!g_cb_open) {
        vkResetCommandBuffer_(g_cb, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer_(g_cb, &bi); g_cb_open = 1; g_cb_ops = 0;
        // the submission before may still run: nothing here starts before it has finished (a barrier's first scope is all earlier work on the queue)
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};
        vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        prof_cb_begin();
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
void ring_forget(uint64_t addr, uint64_t n);   // gfx_vk_draw.inc: guest bytes changed mid-submission
static void tex_fill(vtex *v, const tex_t *t);   // gfx_vk_draw.inc: a texture the GPU has not written gets its guest-memory content

// ---------------------------------------------------------------- operations
static void clear_tex(const p_clear_tex *c) {
    g_rt_hint = 1; vtex *v = use(c->tex); g_rt_hint = 0; if (!v) return;   // a cleared texture is a render target (resolution scaling)
    gpu_wrote(v);
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
    if (v->sc != 1.0f) { static int w; if (!w++) LOG("[vk] a buffer copy into a scaled render target (serial %llu) was skipped", (unsigned long long)c->tex->serial); return; }
    tex_fill(v, c->tex); gpu_wrote(v);   // a copy into part of a texture keeps the rest: its guest content goes in first
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
// A copy from a scaled render target into a buffer (resolution scaling): the region is first scaled back into an image of the guest's size (kept per format and size)
static struct { VkFormat f; uint32_t w, h; VkImage img; VkDeviceMemory m; } g_unsc[16]; static int g_nunsc;
static VkImage unscaled_tmp(VkFormat f, uint32_t w, uint32_t h) {
    for (int i = 0; i < g_nunsc; i++) if (g_unsc[i].f == f && g_unsc[i].w == w && g_unsc[i].h == h) return g_unsc[i].img;
    if (g_nunsc == 16) return VK_NULL_HANDLE;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img; if (vkCreateImage_(VK.dev, &ci, NULL, &img)) return VK_NULL_HANDLE;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements_(VK.dev, img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)}; VkDeviceMemory m;
    if (ai.memoryTypeIndex == ~0u || vkAllocateMemory_(VK.dev, &ai, NULL, &m)) { vkDestroyImage_(VK.dev, img, NULL); return VK_NULL_HANDLE; }
    vkBindImageMemory_(VK.dev, img, m, 0);
    VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {fmt_aspects(f), 0, 1, 0, 1}};
    vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
    g_unsc[g_nunsc++] = (__typeof__(g_unsc[0])){f, w, h, img, m};
    return img;
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
    {   // inside the image: a region past its level's extent or its layers is refused by Vulkan (and reads nothing on the Switch)
        uint32_t gw = v->sc != 1.0f ? v->gw : v->w, gh = v->sc != 1.0f ? v->gh : v->h;   // in the guest's texels (a scaled target is checked at its guest size)
        uint32_t lw = gw >> lv ? gw >> lv : 1, lh = gh >> lv ? gh >> lv : 1;
        if (r->xoffset < 0 || r->yoffset < 0 || (uint32_t)(r->xoffset + r->width) > lw || (uint32_t)(r->yoffset + r->height) > lh || (!v->is3d && l0 + (uint32_t)r->zoffset >= v->layers)) {
            static int w2; if (w2++ < 5) LOG("[vk] a texture → buffer copy outside the texture (%d,%d %dx%d layer %d of a %ux%u level, %u layers) was skipped", r->xoffset, r->yoffset, r->width, r->height, r->zoffset, lw, lh, v->layers);
            return;
        }
    }
    VkImage src_img = v->img; NVNcopyRegion rr = *r; uint32_t src_lv = lv, src_layer = v->is3d ? 0 : l0 + (uint32_t)r->zoffset;
    if (v->sc != 1.0f) {   // scaled: blitted to the guest's size first
        VkImage tmp = unscaled_tmp(v->fmt, (uint32_t)r->width, (uint32_t)r->height); if (!tmp) return;
        VkImageAspectFlags as = fmt_aspects(v->fmt);
        int x0 = (int)(r->xoffset * v->sc), y0 = (int)(r->yoffset * v->sc), x1 = (int)((r->xoffset + r->width) * v->sc + 0.5f), y1 = (int)((r->yoffset + r->height) * v->sc + 0.5f);
        VkImageBlit bl = {{as, lv, src_layer, 1}, {{x0, y0, 0}, {x1 > x0 ? x1 : x0 + 1, y1 > y0 ? y1 : y0 + 1, 1}}, {as, 0, 0, 1}, {{0, 0, 0}, {r->width, r->height, 1}}};
        vkCmdBlitImage_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, fmt_depth(v->fmt) || fmt_int(v->fmt) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR); full_barrier();
        src_img = tmp; rr.xoffset = rr.yoffset = 0; src_lv = 0; src_layer = 0;
    }
    r = &rr;
    stage_t *s = stage_new(wb * rows); if (!s) return;
    s->rb_dst = (uint8_t *)(uintptr_t)dst; s->rb_row = row; s->rb_src_row = wb; s->rb_rows = rows; s->rb_n = wb; s->fwo = -1;
    if (row == wb && (s->orig = malloc(wb * rows))) memcpy(s->orig, s->rb_dst, wb * rows);   // packed: a later submission may get it forwarded on the GPU
    VkBufferImageCopy bic = {0, 0, 0, {fmt_aspects(v->fmt) & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT), src_lv, src_layer, 1},
                             {r->xoffset, r->yoffset, v->is3d && src_img == v->img ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, 1}};
    if (!bic.imageSubresource.aspectMask) bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    vkCmdCopyImageToBuffer_(g_cb, src_img, VK_IMAGE_LAYOUT_GENERAL, s->b, 1, &bic); full_barrier();
}
static void copy_t2t(const p_copy_t2t *c) {
    vtex *s = use(c->src), *d = use(c->dst); if (!s || !d) return;
    tex_fill(s, c->src); tex_fill(d, c->dst); gpu_wrote(d);
    uint32_t slv, snl, sl0, snlay, dlv, dnl, dl0, dnlay;
    view_range(s, &c->sv, c->has_sv, &slv, &snl, &sl0, &snlay); view_range(d, &c->dv, c->has_dv, &dlv, &dnl, &dl0, &dnlay);
    NVNcopyRegion as = c->sr, bs = c->dr; const NVNcopyRegion *a = &as, *b = &bs;
    // scaled render targets (resolution scaling): their regions in image texels
    #define SCR(v, r) do { if ((v)->sc != 1.0f) { (r).xoffset = (int)((r).xoffset * (v)->sc + 0.5f); (r).yoffset = (int)((r).yoffset * (v)->sc + 0.5f); \
        (r).width = (int)((r).width * (v)->sc + 0.5f); (r).height = (int)((r).height * (v)->sc + 0.5f); if ((r).width < 1) (r).width = 1; if ((r).height < 1) (r).height = 1; } } while (0)
    SCR(s, as); SCR(d, bs);
    #undef SCR
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
// Where a submission's time goes (logged with the replay summary): recording (the command stream walked: descriptor writes, ring copies, uploads), the submit call,
// the wait for the GPU, and the work after it (read-backs, write-backs, resets)
static uint64_t g_t_rec, g_t_submit, g_t_wait, g_t_done, g_t_n; static int g_exec_depth; static uint64_t g_exec_t0;
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static void mtl_exec_body(const uint8_t *p);
// The submission being recorded belongs to one thread at a time: from mtl_sub_begin (or the first mtl_exec) to mtl_sub_commit. Others that need the recorded work on the
// GPU (a fence, a wait, a present, a texture's release) take it too and hand the batch over (sub_flush)
static pthread_mutex_t g_rec_mu = PTHREAD_MUTEX_INITIALIZER; static __thread int t_rec_held;
static uint64_t g_early_syncs; static _Atomic uint64_t g_wait_sync_ns, g_wait_acq_ns, g_wait_flush_ns, g_wait_gpu_ns, g_wait_gpu_n, g_wait_unsub;   // how long the game blocked for the GPU in sync waits (an acquired texture's sync apart)
static int g_batch = -1, g_deferred; static uint64_t g_batched, g_gpu_subs, g_flush_why[8];
enum { FL_COMMIT, FL_PRESENT, FL_FENCE, FL_WAIT, FL_RELEASE, FL_MAX, FL_NEED };
static const char *const k_flush[] = {"a game submission that could not wait", "a present", "a fence", "a wait", "a texture release", "four batched", "other"};
static void rec_take(void) { if (!t_rec_held) { pthread_mutex_lock(&g_rec_mu); t_rec_held = 1; } }
static void rec_give(void) { if (t_rec_held) { t_rec_held = 0; pthread_mutex_unlock(&g_rec_mu); } }
void mtl_exec(const uint8_t *p) {
    if (!g_exec_depth) rec_take();
    if (!g_exec_depth++) g_exec_t0 = mono_ns();
    mtl_exec_body(p);
    if (!--g_exec_depth) g_t_rec += mono_ns() - g_exec_t0;
}
static void mtl_exec_body(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        g_cur_op = h.op;
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); mtl_exec((const uint8_t *)(uintptr_t)t); break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); post((post_op){1, val, m, v, 0, 0}); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); post((post_op){2, NULL, 0, 0, ad, (uint32_t)t}); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = atomic_load(&g_fence) + 1; s->fenced = 1; break; }   // this submission's number when it is handed over
        case OP_CLEAR_BUF: { const p_clear_buf *c = (const p_clear_buf *)a; pend_hazard(c->addr, c->size); uint32_t *d = (uint32_t *)(uintptr_t)c->addr; for (uint64_t i = 0; i < c->size / 4; i++) d[i] = c->value; ring_forget(c->addr, c->size); break; }
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); pend_hazard(s_, n); pend_hazard(d, n); memmove((void *)(uintptr_t)d, (const void *)(uintptr_t)s_, n); ring_forget(d, n); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); pend_hazard(d, n); memcpy((void *)(uintptr_t)d, a + 16, n); ring_forget(d, n); break; }
        case OP_COPY_STRIDE: { const p_copy_stride *cs = (const p_copy_stride *)a; if (cs->image) R.copy_img_stride = cs->stride; else R.copy_stride = cs->stride; break; }
        case OP_SET_RT: if (VK.ok && P.open) pass_end(); R.rt = *(const p_set_rt *)a; R.has_rt = 1; break;   // waiting clears stay waiting: the next pass takes them, or does them first
        case OP_CLEAR_COLOR: if (VK.ok) clear_color((const p_clear_color *)a); break;   // into the render pass when it can (gfx_vk_draw.inc)
        case OP_CLEAR_DS: if (VK.ok) clear_ds((const p_clear_ds *)a); break;
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
    rec_take();
    // the window comes up with the first submission, so the game's textures have a device from the start
    if (vkw_wanted() && !VK.ok) { vkw_init(); const char *e = getenv("DM_VK_DRAW"); g_draw_on = !(e && *e == '0'); if (VK.ok) LOG("[vk] draws: %s", g_draw_on ? "on (stage C)" : "off (DM_VK_DRAW=0)"); }
}
// ---------------------------------------------------------------- submissions in flight: three slots. The submitting thread records one while the GPU runs the others (up to
// two, DM_VK_INFLIGHT=1 keeps one); at a commit the recorded one is handed to the GPU and swaps with a completed slot, waiting for the oldest when none is. With one in
// flight the GPU went idle whenever the game did its own work between submissions (about a third of each frame): a second one keeps it fed. A completion thread waits
// for each submission's fence in order and lands what the game sees of it (read-backs, storage buffer write-backs, events, counters) as soon as the GPU is done, so a
// game polling an event never waits for its next submission. Vulkan objects (staging buffers, buffer views, descriptor pools, the command buffer) are reset on the
// submitting thread
typedef struct {
    uint64_t id; VkCommandBuffer cb; VkFence fence;
    stage_t *stage; int nstage, stagecap;
    post_op *post; int npost, postcap;
    vtex **lw; int nlw, clw; uint64_t *lws; int clws;   // lws: the linear textures' serials (one released meanwhile is left alone)
    wback *wb; int nwb, cwb;
    VkBufferView *tbv; int ntbv, ctbv;
    VkDescriptorPool *dp; int ndp;
    VkQueryPool pq; uint32_t npq; pmark *pm; int npm;
    uint64_t ring_base, t_handed; VkCommandBuffer pre;
    int idx, queued;   // queued: handed over, its completion not started (the completion thread's mark)
    int dirty;   // completed, its Vulkan objects not reset yet
    int failed;  // vkQueueSubmit refused it: no fence to wait for
    int cpu_out; // it writes guest memory when it completes: 1 storage write-backs, 3 texture-to-buffer copies, 2 only linear textures' read-backs (a sync covering it
                 // waits for the GPU, as DM_VK_EARLY_SYNC says)
} subslot;
static subslot g_sl[NSUB - 1] = {{.dp = g_dp_store[1], .pm = g_pm_store[1], .ring_base = RING_PART, .idx = 0}, {.dp = g_dp_store[2], .pm = g_pm_store[2], .ring_base = 2 * RING_PART, .idx = 1}};
static pthread_mutex_t g_cm_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cm_done, g_cm_work; static int g_cm_started, g_async = -1, g_inflight = 2;
static uint64_t g_t_gpu, g_t_land, g_t_wake, g_t_post, g_n_post;   // the completion thread's waits for the GPU (how long submissions ran after their hand-over)
static void cm_init(void) {
    pthread_condattr_t ca; pthread_condattr_init(&ca); pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
    pthread_cond_init(&g_cm_done, &ca); pthread_cond_init(&g_cm_work, NULL); pthread_condattr_destroy(&ca);
}
static pthread_once_t g_cm_once = PTHREAD_ONCE_INIT;
static int any_busy(void) { for (int i = 0; i < NSUB - 1; i++) if (atomic_load(&g_pr[i].busy)) return 1; return 0; }
static void pend_wait_slot(int i) {   // that submission has completed when this returns (and every one before it: they complete in order)
    if (!atomic_load(&g_pr[i].busy)) return;
    pthread_once(&g_cm_once, cm_init);
    pthread_mutex_lock(&g_cm_mu); while (atomic_load(&g_pr[i].busy)) pthread_cond_wait(&g_cm_done, &g_cm_mu); pthread_mutex_unlock(&g_cm_mu);
}
static void pend_wait(void) {   // every submission in flight has completed when this returns
    if (!any_busy()) return;
    pthread_once(&g_cm_once, cm_init);
    pthread_mutex_lock(&g_cm_mu); while (any_busy()) pthread_cond_wait(&g_cm_done, &g_cm_mu); pthread_mutex_unlock(&g_cm_mu);
}
static void flush_from(int why);
static void sub_wait_all(void) { flush_from(FL_RELEASE); pend_wait(); }
// What the game sees of a completed submission, on the completion thread (guest pages the dirty tracking protects are opened first)
static void slot_land(subslot *s) {
    if (s->cb && !s->failed) {
        uint64_t t0 = mono_ns();
        VkResult w = vkWaitForFences_(VK.dev, 1, &s->fence, VK_TRUE, 5000000000ull);
        if (w) { LOG("[RED] [vk] the GPU did not finish a submission in 5 s (%d): device lost or hung", w); vkDeviceWaitIdle_(VK.dev); }
        vkResetFences_(VK.dev, 1, &s->fence);
        g_t_gpu += mono_ns() - t0;
    }
    uint64_t tl = mono_ns();
    prof_collect(s->pq, s->npq, s->pm, s->npm);
    for (int i = 0; i < s->nstage; i++) {
        stage_t *st = &s->stage[i];
        if (st->rb_dst) { dm_guest_write_prepare(st->rb_dst, (st->rb_rows - 1) * st->rb_row + st->rb_n); for (size_t r = 0; r < st->rb_rows; r++) memcpy(st->rb_dst + r * st->rb_row, (uint8_t *)st->p + r * st->rb_src_row, st->rb_n); }
        if (st->dump_fmt) dump_write(st);
    }
    for (int i = 0; i < s->nwb; i++) wb_land(&s->wb[i]);
    // its events and counters, in record order
    uint64_t now = gpu_ticks(), tp = mono_ns(); g_n_post += s->npost;
    for (int i = 0; i < s->npost; i++) {
        post_op *o = &s->post[i];
        if (o->kind == 1) { dm_guest_write_prepare((void *)o->val, 4); if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else {
            // samples passed (occlusion queries): every query is reported visible rather than culling everything; timestamps: the guest clock in GPU ticks
            uint64_t rec[2] = {o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? now : 0, now};
            dm_guest_write_prepare((void *)(uintptr_t)o->addr, sizeof rec); memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
        }
    }
    uint64_t te = mono_ns(); g_t_post += te - tp; g_t_land += te - tl;
}
static void *cm_thread(void *u) {
    (void)u;
    for (;;) {
        subslot *s = NULL;
        pthread_mutex_lock(&g_cm_mu);
        for (;;) {   // the oldest submission handed over and not landed yet
            for (int i = 0; i < NSUB - 1; i++) if (g_sl[i].queued && (!s || g_sl[i].id < s->id)) s = &g_sl[i];
            if (s) break;
            pthread_cond_wait(&g_cm_work, &g_cm_mu);
        }
        s->queued = 0;
        pthread_mutex_unlock(&g_cm_mu);
        g_t_wake += mono_ns() - s->t_handed;
        slot_land(s);
        pthread_mutex_lock(&g_cm_mu); atomic_store(&g_done, s->id); atomic_store(&g_pr[s->idx].busy, 0); pthread_cond_broadcast(&g_cm_done); pthread_mutex_unlock(&g_cm_mu);
    }
    return NULL;
}
// A completed slot's Vulkan objects reset for its next submission (the submitting thread), and its linear textures' storage is their content again
static void slot_clean(subslot *s) {
    if (!s->dirty) return;
    s->dirty = 0;
    for (int i = 0; i < s->nstage; i++) { stage_t *st = &s->stage[i]; vkUnmapMemory_(VK.dev, st->m); vkDestroyBuffer_(VK.dev, st->b, NULL); vkFreeMemory_(VK.dev, st->m, NULL); free(st->orig); }
    s->nstage = 0;
    if (s->nlw) {
        pthread_mutex_lock(&g_tex_mu);
        for (int i = 0; i < s->nlw; i++) { vtex *v = s->lw[i]; if (v->serial == s->lws[i] && v->img && !v->wrote) { v->uploaded = 1; track(v); } }
        pthread_mutex_unlock(&g_tex_mu);
        s->nlw = 0;
    }
    for (int i = 0; i < s->ntbv; i++) vkDestroyBufferView_(VK.dev, s->tbv[i], NULL);
    s->ntbv = 0;
    for (int i = 0; i < s->ndp; i++) vkResetDescriptorPool_(VK.dev, s->dp[i], 0);
    for (int i = 0; i < s->nwb; i++) free(s->wb[i].orig);
    s->npost = 0; s->nwb = 0; s->npq = 0; s->npm = 0;
}
// A slot for the recorded submission to swap with: one not in flight, with fewer than g_inflight in flight (else the oldest is waited for)
static subslot *slot_free(void) {
    for (;;) {
        int nb = 0; subslot *fr = NULL, *old = NULL;
        for (int i = 0; i < NSUB - 1; i++) {
            if (atomic_load(&g_pr[i].busy)) { nb++; if (!old || g_sl[i].id < old->id) old = &g_sl[i]; } else if (!fr) fr = &g_sl[i];
        }
        if (fr && nb < g_inflight) return fr;
        pend_wait_slot(old->idx);
    }
}
// Forwarding (gfx_vk_draw.inc, the forwarding buffer): the bytes [addr, addr + size) wanted in the ring at ring_off are a storage buffer result of a submission in flight,
// copied to the forwarding buffer at its end. When the guest's bytes are still those it recorded (the CPU has not written them since), a GPU copy before this submission
// puts the result in the ring and nothing waits. 0: not forwarded (the caller waits as before)
static int fwd_try(uint64_t addr, uint64_t size, uint64_t ring_off) {
    // what the submissions in flight write into these bytes: a storage write-back or a packed texture-to-buffer copy holding all of them (the newest), and counters or events
    VkBuffer src = VK_NULL_HANDLE; uint64_t src_off = 0, src_id = 0; int posts = 0;
    for (int s = 0; s < NSUB - 1; s++) {
        if (!atomic_load_explicit(&g_pr[s].busy, memory_order_acquire) || addr >= g_pr[s].hi || addr + size <= g_pr[s].lo) continue;
        subslot *sl = &g_sl[s];
        #define FW_HOLD(A, N, FWO, EP, ORIG) do { uint64_t a_ = (A), n_ = (N); if (addr >= a_ + n_ || a_ >= addr + size) break; \
            if (!(a_ <= addr && addr + size <= a_ + n_) || (FWO) < 0 || (EP) != FW.epoch || !(ORIG)) { g_fwd_miss++; return 0; } \
            if (memcmp((const void *)(uintptr_t)addr, (ORIG) + (addr - a_), size)) { g_fwd_miss++; return 0; }   /* the CPU wrote them since: its bytes are the newer ones */ \
            if (!src || sl->id > src_id) { src = FW.b; src_off = (uint64_t)(FWO) + (addr - a_); src_id = sl->id; } } while (0)
        for (int i = 0; i < sl->nwb; i++) FW_HOLD(sl->wb[i].addr, sl->wb[i].size, sl->wb[i].fwo, sl->wb[i].fw_epoch, sl->wb[i].orig);
        for (int i = 0; i < sl->nstage; i++) { stage_t *st = &sl->stage[i]; if (st->rb_dst) FW_HOLD((uint64_t)(uintptr_t)st->rb_dst, (st->rb_rows - 1) * st->rb_row + st->rb_n, st->fwo, st->fw_epoch, st->orig); }
        #undef FW_HOLD
        for (int i = 0; i < sl->npost; i++) {
            const post_op *o = &sl->post[i]; uint64_t a_ = o->kind == 1 ? (uint64_t)(uintptr_t)o->val : o->addr, n_ = o->kind == 1 ? 4 : 16;
            if (addr < a_ + n_ && a_ < addr + size) { if (o->kind == 1 && o->mode == 1) return 0; posts = 1; }   // an event add waits (it would count twice)
        }
    }
    if (src && posts) return 0;
    if (posts) {   // counters and events: their values are known now (as when a sync is done early): written into guest memory, the ring copy takes them
        pthread_mutex_lock(&g_cm_mu);
        uint64_t now = gpu_ticks();
        for (int s = 0; s < NSUB - 1; s++) {
            subslot *sl = &g_sl[s]; if (!atomic_load(&g_pr[s].busy)) continue;
            for (int i = 0; i < sl->npost; i++) {
                post_op *o = &sl->post[i]; uint64_t a_ = o->kind == 1 ? (uint64_t)(uintptr_t)o->val : o->addr, n_ = o->kind == 1 ? 4 : 16;
                if (!(addr < a_ + n_ && a_ < addr + size)) continue;
                if (o->kind == 1) { dm_guest_write_prepare((void *)o->val, 4); *o->val = o->v; }
                else { uint64_t rec[2] = {o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? now : 0, now};
                       dm_guest_write_prepare((void *)(uintptr_t)o->addr, sizeof rec); memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
            }
        }
        pthread_mutex_unlock(&g_cm_mu);
        g_fwd_post++;
        return 1;
    }
    if (!src) return 0;
    if (!g_pre_open) {
        if (!g_pre) { VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; if (vkAllocateCommandBuffers_(VK.dev, &cai, &g_pre)) { g_pre = VK_NULL_HANDLE; return 0; } }
        vkResetCommandBuffer_(g_pre, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer_(g_pre, &bi); g_pre_open = 1;
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};
        vkCmdPipelineBarrier_(g_pre, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    VkBufferCopy bc = {src_off, ring_off, size}; vkCmdCopyBuffer_(g_pre, src, RG.b, 1, &bc);
    static int said; g_fwd_n++; if (!said++) LOG("[vk] first storage result forwarded on the GPU (%llu bytes at %#llx)", (unsigned long long)size, (unsigned long long)addr);
    return 1;
}
#define SWP(a, b) do { __typeof__(a) t_ = (a); (a) = (b); (b) = t_; } while (0)
// The recorded submission (submitted already) goes into slot P, in flight; P's completed (and cleaned) things become the recording ones
static void slot_handover(subslot *P, uint64_t id, int failed) {
    SWP(g_cb, P->cb); SWP(g_fence_vk, P->fence); SWP(g_pre, P->pre);
    SWP(g_stage, P->stage); SWP(g_nstage, P->nstage); SWP(g_stagecap, P->stagecap);
    SWP(g_post, P->post); SWP(g_npost, P->npost); SWP(g_postcap, P->postcap);
    SWP(g_lw, P->lw); SWP(g_nlw, P->nlw); SWP(g_clw, P->clw);
    SWP(g_wb, P->wb); SWP(g_nwb, P->nwb); SWP(g_cwb, P->cwb);
    SWP(g_tbv, P->tbv); SWP(g_ntbv, P->ntbv); SWP(g_ctbv, P->ctbv);
    SWP(g_dp, P->dp); SWP(g_ndp, P->ndp);
    SWP(g_pq, P->pq); SWP(g_npq, P->npq); SWP(g_pm, P->pm); SWP(g_npm, P->npm);
    SWP(RG.base, P->ring_base);   // the GPU reads its ring part until it completes: the recording goes on in the completed slot's part
    P->id = id; P->dirty = 1; P->failed = failed;
    P->cpu_out = P->nwb > 0; for (int i = 0; i < P->nstage && !P->cpu_out; i++) if (P->stage[i].rb_dst && !P->stage[i].lin_rb) P->cpu_out = 3;
    if (P->nwb) P->cpu_out = 1; else if (!P->cpu_out && P->nlw) P->cpu_out = 2;
    // linear textures it reads back: written again by a later submission, they are read back with that one too
    if (P->nlw > P->clws) { P->clws = P->clw; P->lws = realloc(P->lws, P->clws * sizeof *P->lws); }
    for (int i = 0; i < P->nlw; i++) { P->lws[i] = P->lw[i]->serial; P->lw[i]->wrote = 0; }
    // the guest bytes it writes when it completes
    __typeof__(g_pr[0]) *R = &g_pr[P->idx];
    R->n = 0; R->lo = ~0ull; R->hi = 0;
    #define PR(A, N) do { uint64_t a_ = (A), n_ = (N); if (!a_ || !n_) break; if (R->n == R->cap) { R->cap = R->cap ? 2 * R->cap : 64; R->r = realloc(R->r, R->cap * sizeof *R->r); } \
        R->r[R->n++] = (grange){a_, a_ + n_}; if (a_ < R->lo) R->lo = a_; if (a_ + n_ > R->hi) R->hi = a_ + n_; } while (0)
    for (int i = 0; i < P->nstage; i++) if (P->stage[i].rb_dst) PR((uint64_t)(uintptr_t)P->stage[i].rb_dst, (P->stage[i].rb_rows - 1) * P->stage[i].rb_row + P->stage[i].rb_n);
    for (int i = 0; i < P->nwb; i++) PR(P->wb[i].addr, P->wb[i].size);
    for (int i = 0; i < P->npost; i++) { if (P->post[i].kind == 1) PR((uint64_t)(uintptr_t)P->post[i].val, 4); else PR(P->post[i].addr, 16); }
    #undef PR
    rec_reset();
    if (!g_cm_started) { pthread_once(&g_cm_once, cm_init); pthread_t th; pthread_create(&th, NULL, cm_thread, NULL); pthread_detach(th); g_cm_started = 1; }
    P->t_handed = mono_ns();
    pthread_mutex_lock(&g_cm_mu); atomic_store(&R->busy, 1); P->queued = 1; pthread_cond_signal(&g_cm_work); pthread_mutex_unlock(&g_cm_mu);
}
// Game submissions batched into one GPU submission: each GPU submission cost about 2.5 ms of GPU time before it started (on the Thor, in Game Mode and on the desktop
// alike), three a frame. A game submission stays in the open command buffer when nothing in it needs landing in guest memory soon (no event, no read-back or write-back),
// and goes to the GPU with the next one that does, a present, a fence or wait, a texture's release, or after a few. DM_VK_BATCH=0 hands each one over
static int batch_ok(void) {
    if (!g_batch || g_dump_subs || g_deferred >= 3 || g_nlw || g_nwb) return 0;
    for (int i = 0; i < g_npost; i++) if (g_post[i].kind == 1) return 0;   // an event: a game may poll it
    for (int i = 0; i < g_nstage; i++) if (g_stage[i].rb_dst || g_stage[i].dump_fmt) return 0;
    return RG.used - RG.base < RING_PART / 2;
}
static uint64_t sub_commit_locked(int force);
static void sub_flush(int why) {   // the batched game submissions to the GPU (the caller holds the recording)
    if (g_cb_open && g_deferred) { g_flush_why[why]++; sub_commit_locked(1); }
}
static void flush_from(int why) { int mine = t_rec_held; if (!mine) pthread_mutex_lock(&g_rec_mu); sub_flush(why); if (!mine) pthread_mutex_unlock(&g_rec_mu); }
uint64_t mtl_sub_commit(void) { uint64_t f = sub_commit_locked(0); rec_give(); return f; }
static uint64_t sub_commit_locked(int force) {
    if (g_batch < 0) { const char *e = getenv("DM_VK_BATCH"); g_batch = !(e && *e == '0'); LOG("[vk] game submissions %s", g_batch ? "batched into GPU submissions (DM_VK_BATCH=0: one each)" : "each a GPU submission (DM_VK_BATCH=0)"); }
    if (!force && g_cb_open && batch_ok()) { g_deferred++; g_batched++; return atomic_load(&g_fence) + 1; }
    if (!force && g_deferred) g_flush_why[g_deferred >= 3 ? FL_MAX : FL_COMMIT]++;
    g_deferred = 0;
    if (g_async < 0) {
        const char *e = getenv("DM_VK_ASYNC"); g_async = !(e && *e == '0');
        const char *n = getenv("DM_VK_INFLIGHT"); if (n && (*n == '1' || *n == '2')) g_inflight = *n - '0';
        if (g_async) LOG("[vk] submissions: recorded while up to %d before run on the GPU (DM_VK_INFLIGHT=1 or 2; DM_VK_ASYNC=0 waits for each)", g_inflight);
        else LOG("[vk] submissions: each waited for (DM_VK_ASYNC=0)");
    }
    uint64_t f;
    if (g_cb_open) {
        uint64_t t0 = mono_ns();
        pass_end();
        fw_copy_out();
        linear_readbacks();
        prof_cb_end();
        vkEndCommandBuffer_(g_cb); g_cb_open = 0;
        VkCommandBuffer cbs[2] = {g_pre, g_cb}; int npre = g_pre_open;
        if (g_pre_open) { vkEndCommandBuffer_(g_pre); g_pre_open = 0; }
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, npre ? 2 : 1, npre ? cbs : &g_cb, 0, NULL};
        pthread_mutex_lock(&VK.qmu);
        VkResult r = vkQueueSubmit_(VK.q, 1, &si, g_fence_vk);
        pthread_mutex_unlock(&VK.qmu);
        if (r) LOG("[RED] [vk] submission failed (%d)", r);
        uint64_t t1 = mono_ns(); g_t_submit += t1 - t0;
        subslot *P = slot_free();   // usually completed long ago: it ran while this one was recorded
        uint64_t t2 = mono_ns(); g_t_wait += t2 - t1;
        slot_clean(P);
        f = atomic_load(&g_fence) + 1;
        slot_handover(P, f, r != VK_SUCCESS);
        atomic_store(&g_fence, f); g_gpu_subs++;
        if (!g_async) pend_wait();
        g_t_done += mono_ns() - t2; g_t_n++;
    } else {
        // nothing for the GPU: its events and counters land after the submissions before it, now
        pend_wait();
        f = atomic_load(&g_fence) + 1;
        uint64_t now = gpu_ticks();
        for (int i = 0; i < g_npost; i++) {
            post_op *o = &g_post[i];
            if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
            else { uint64_t rec[2] = {o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? now : 0, now}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
        }
        g_npost = 0;
        pthread_mutex_lock(&g_cm_mu); atomic_store(&g_fence, f); atomic_store(&g_done, f); pthread_once(&g_cm_once, cm_init); pthread_cond_broadcast(&g_cm_done); pthread_mutex_unlock(&g_cm_mu);
    }
    dump_submitted();
    static uint64_t reported;
    if (f == 1 || f - reported >= 600) {
        reported = f; uint64_t np = 0, nm = 0, nf = 0; vks_stats(&np, &nm, &nf);
        char ds[640]; draw_summary(ds, sizeof ds);
        // memory: the textures alive on the GPU side, this process's resident memory, and what the system has left (a game's memory file fails with SIGBUS when it runs out)
        long rss_kb = 0, avail_kb = 0, swapf_kb = 0; char ln[256];
        FILE *fs = fopen("/proc/self/status", "r"); if (fs) { while (fgets(ln, sizeof ln, fs)) sscanf(ln, "VmRSS: %ld", &rss_kb); fclose(fs); }
        FILE *fm = fopen("/proc/meminfo", "r"); if (fm) { while (fgets(ln, sizeof ln, fm)) { sscanf(ln, "MemAvailable: %ld", &avail_kb); sscanf(ln, "SwapFree: %ld", &swapf_kb); } fclose(fm); }
        LOG("[vk] memory: textures alive %llu (%.0f MB), process resident %.2f GB, system available %.2f GB, swap free %.2f GB", (unsigned long long)g_tex_live, g_tex_live_bytes / 1048576.0,
            rss_kb / 1048576.0, avail_kb / 1048576.0, swapf_kb / 1048576.0);
        LOG("[vk] replay: %llu submissions; draws %llu (%llu with their shaders ready), dispatches %llu (%llu ready); programs %llu, shaders missing %llu, refused %llu; %s",
            (unsigned long long)f, (unsigned long long)g_draws, (unsigned long long)g_draws_ready, (unsigned long long)g_dispatches, (unsigned long long)g_disp_ready,
            (unsigned long long)np, (unsigned long long)nm, (unsigned long long)nf, ds);
        if (g_t_n) LOG("[vk] time per submission (last %llu): recording %.2f ms, submit %.2f ms, waiting for the one before %.2f ms, after %.2f ms; GPU still busy at hand-over %.2f ms; "
                       "waits for a write-back in flight %llu; storage write-backs %.2f MB per submission, %.3f MB of it changed; ring %.2f MB per submission",
            (unsigned long long)g_t_n, g_t_rec / 1e6 / g_t_n, g_t_submit / 1e6 / g_t_n, g_t_wait / 1e6 / g_t_n, g_t_done / 1e6 / g_t_n, g_t_gpu / 1e6 / g_t_n,
            (unsigned long long)g_hazard_waits, g_wb_bytes / 1048576.0 / g_t_n, g_wb_changed / 1048576.0 / g_t_n, ring_mb_per_sub(g_t_n));
        if (g_gpu_subs) { char fw[400]; int o = 0; fw[0] = 0; for (int i = 0; i < 7; i++) if (g_flush_why[i] && o < 380) o += snprintf(fw + o, sizeof fw - o, "%s%llu at %s", o ? ", " : "", (unsigned long long)g_flush_why[i], k_flush[i]);
            static uint64_t tlast; uint64_t tn = mono_ns(); double secs = tlast ? (tn - tlast) / 1e9 : 0; tlast = tn;
            LOG("[vk] batching: %.2f game submissions per GPU submission; batches handed over at: %s; the game blocked for the GPU %.0f ms a second in sync waits, %.0f in acquired-texture syncs; syncs done early %llu; of the blocking: handing batched work over %.0f ms, waiting for the GPU %.0f ms (%llu waits); syncs not handed over yet %llu",
                (double)(g_gpu_subs + g_batched) / g_gpu_subs, o ? fw : "none", secs > 0 ? atomic_exchange(&g_wait_sync_ns, 0) / 1e6 / secs : 0.0, secs > 0 ? atomic_exchange(&g_wait_acq_ns, 0) / 1e6 / secs : 0.0, (unsigned long long)g_early_syncs,
                secs > 0 ? atomic_exchange(&g_wait_flush_ns, 0) / 1e6 / secs : 0.0, secs > 0 ? atomic_exchange(&g_wait_gpu_ns, 0) / 1e6 / secs : 0.0, (unsigned long long)atomic_exchange(&g_wait_gpu_n, 0), (unsigned long long)atomic_exchange(&g_wait_unsub, 0)); g_early_syncs = 0;
            g_gpu_subs = g_batched = 0; memset(g_flush_why, 0, sizeof g_flush_why); }
        if (g_t_n) LOG("[vk] completion per submission: thread wake-up %.2f ms, after the fence %.2f ms (events and counters %.2f ms, %.0f of them)", g_t_wake / 1e6 / g_t_n, g_t_land / 1e6 / g_t_n,
                       g_t_post / 1e6 / g_t_n, (double)g_n_post / g_t_n);
        g_t_wake = g_t_land = g_t_post = g_n_post = 0;
        g_wb_bytes = g_wb_changed = 0;
        if (g_hz[0].line) { char hb[400]; int o = 0; for (int i = 0; i < 16 && g_hz[i].line && o < 380; i++) o += snprintf(hb + o, sizeof hb - o, "%s%s:%d: %llu waits, %.1f ms, %llu bytes read each", i ? "; " : "", strrchr(g_hz[i].file, '/') ? strrchr(g_hz[i].file, '/') + 1 : g_hz[i].file, g_hz[i].line,
                (unsigned long long)g_hz[i].n, g_hz[i].ns / 1e6, (unsigned long long)(g_hz[i].bytes / g_hz[i].n));
            LOG("[vk] waits for a write-back in flight, by where the bytes are read: %s", hb); memset(g_hz, 0, sizeof g_hz); }
        if (g_dsc_hit + g_dsc_miss) { LOG("[vk] descriptor sets: %.0f%% reused (%llu reused, %llu written)", 100.0 * g_dsc_hit / (g_dsc_hit + g_dsc_miss), (unsigned long long)g_dsc_hit, (unsigned long long)g_dsc_miss); g_dsc_hit = g_dsc_miss = 0; }
        if (g_fwd_n || g_fwd_miss || g_fwd_post) { LOG("[vk] results in flight forwarded: on the GPU %llu times, counters and events written at once %llu times; %llu could not be (the CPU had written them, partly covered, or the forwarding buffer had moved on)",
            (unsigned long long)g_fwd_n, (unsigned long long)g_fwd_post, (unsigned long long)g_fwd_miss); g_fwd_n = g_fwd_miss = g_fwd_post = 0; }
        g_t_rec = g_t_submit = g_t_wait = g_t_done = g_t_n = g_t_gpu = 0; g_hazard_waits = 0;
        prof_report();
    }
    return f;
}
uint64_t mtl_last_fence(void) { if (g_deferred) flush_from(FL_FENCE); return atomic_load(&g_fence); }   // a fence covers everything submitted: batched work goes first
// Completed up to fence: waited for (up to timeout_ns) when that submission was handed to the GPU; one still being recorded is not waited for, as before
// A sync whose submissions the GPU has not finished: done already, when none of them writes guest memory at its end. The replay takes everything a submission reads from
// guest memory at the time it is recorded (ring copies, texture uploads), so the game may reuse its buffers as soon as the work is handed over; what it may read back
// (events, counters) is written now, and again when the GPU is done. The game waited for each frame's sync, so the CPU and GPU took turns (Mario Tennis Fever blocked
// about 600 ms a second). DM_VK_EARLY_SYNC=0 waits for the GPU
static int early_done(uint64_t fence) {
    // DM_VK_EARLY_SYNC: 0 waits for the GPU, 1 (default) early unless results are read back, 2 early also past linear textures' read-backs (the game then may read
    // a linear texture's content of a frame before)
    // 3 also past texture-to-buffer copies (the game may read the copy of a frame before), 4 also past storage buffer write-backs (the same)
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_EARLY_SYNC"); on = e && *e >= '0' && *e <= '4' ? *e - '0' : 1;
        if (on) LOG("[vk] syncs: done once their work is handed to the GPU, unless it reads results back%s (DM_VK_EARLY_SYNC=0 waits for the GPU, 2 also past linear textures, 3 also past texture copies)",
                    on == 4 ? " (none count)" : on == 3 ? " (only storage write-backs count)" : on == 2 ? " (not counting linear textures)" : ""); }
    if (!on || fence > atomic_load(&g_fence)) return 0;
    pthread_once(&g_cm_once, cm_init);
    pthread_mutex_lock(&g_cm_mu);
    int ok = 1;
    for (int i = 0; i < NSUB - 1; i++) if (atomic_load(&g_pr[i].busy) && g_sl[i].id <= fence && ((g_sl[i].cpu_out == 1 && on < 4) || (g_sl[i].cpu_out == 2 && on < 2) || (g_sl[i].cpu_out == 3 && on < 3))) ok = 0;
    if (ok) {
        uint64_t now = gpu_ticks();
        for (int i = 0; i < NSUB - 1; i++) {
            subslot *sl = &g_sl[i]; if (!atomic_load(&g_pr[i].busy) || sl->id > fence) continue;
            for (int k = 0; k < sl->npost; k++) {
                post_op *o = &sl->post[k];
                if (o->kind == 1) { dm_guest_write_prepare((void *)o->val, 4); if (o->mode == 1) continue; *o->val = o->v; }   // an add waits for the GPU's end (it would count twice)
                else { uint64_t rec[2] = {o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->type == 0 || o->type == 0x10) && !DM_GAME_RENDER_COND ? now : 0, now};
                       dm_guest_write_prepare((void *)(uintptr_t)o->addr, sizeof rec); memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
            }
        }
        g_early_syncs++;
    }
    pthread_mutex_unlock(&g_cm_mu);
    return ok;
}
static int wait_timed(uint64_t fence, uint64_t timeout_ns) {
    if (fence <= atomic_load(&g_done)) return 1;
    if (fence > atomic_load(&g_fence) && g_deferred) {   // batched work: handed over first (a poll does it only when the recording is free)
        if (timeout_ns) { uint64_t t0 = mono_ns(); flush_from(FL_WAIT); atomic_fetch_add(&g_wait_flush_ns, mono_ns() - t0); }
        else if (t_rec_held) sub_flush(FL_WAIT);
        else if (!pthread_mutex_trylock(&g_rec_mu)) { sub_flush(FL_WAIT); pthread_mutex_unlock(&g_rec_mu); }
    }
    if (early_done(fence)) return 1;
    if (timeout_ns && fence > atomic_load(&g_fence)) atomic_fetch_add(&g_wait_unsub, 1);
    if (!timeout_ns || fence > atomic_load(&g_fence)) return 0;
    static _Atomic int nlog;
    int nl = atomic_fetch_add(&nlog, 1);
    if (nl < 4 || nl % 500 == 0) {   // why this sync waits for the GPU (the first few, then one in 500)
        char sl[256]; int o = 0;
        for (int i = 0; i < NSUB - 1; i++) o += snprintf(sl + o, sizeof sl - o, " [slot %d: submission %llu, %s, read-backs kind %d, %d stages, %d write-backs, %d linear]", i, (unsigned long long)g_sl[i].id,
                                                          atomic_load(&g_pr[i].busy) ? "in flight" : "done", g_sl[i].cpu_out, g_sl[i].nstage, g_sl[i].nwb, g_sl[i].nlw);
        LOG("[vk] a sync waits for the GPU: fence %llu, handed over %llu, completed %llu, early mode %s;%s", (unsigned long long)fence, (unsigned long long)atomic_load(&g_fence),
            (unsigned long long)atomic_load(&g_done), getenv("DM_VK_EARLY_SYNC") ? getenv("DM_VK_EARLY_SYNC") : "default", sl);
    }
    uint64_t tg = mono_ns();
    pthread_once(&g_cm_once, cm_init);
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t lim = timeout_ns > 3600ull * 1000000000ull ? 3600ull * 1000000000ull : timeout_ns, ns = (uint64_t)ts.tv_nsec + lim % 1000000000ull;
    ts.tv_sec += (time_t)(lim / 1000000000ull + ns / 1000000000ull); ts.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_cm_mu);
    while (fence > atomic_load(&g_done)) if (pthread_cond_timedwait(&g_cm_done, &g_cm_mu, &ts)) break;
    pthread_mutex_unlock(&g_cm_mu);
    atomic_fetch_add(&g_wait_gpu_ns, mono_ns() - tg); atomic_fetch_add(&g_wait_gpu_n, 1);
    return fence <= atomic_load(&g_done);
}
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) {
    if (!timeout_ns || fence <= atomic_load(&g_done)) return mtl_wait(fence, timeout_ns);
    uint64_t t0 = mono_ns(); int r = mtl_wait(fence, timeout_ns); atomic_fetch_add(&g_wait_acq_ns, mono_ns() - t0); return r;
}
static int wait_timed(uint64_t fence, uint64_t timeout_ns);
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (!timeout_ns || fence <= atomic_load(&g_done)) return wait_timed(fence, timeout_ns);
    uint64_t t0 = mono_ns(); int r = wait_timed(fence, timeout_ns); atomic_fetch_add(&g_wait_sync_ns, mono_ns() - t0); return r;
}
void mtl_drain(void) { sub_wait_all(); }
void mtl_finish(void) { sub_wait_all(); }

// The image to show for a present (gfx_null.c mtl_present): the swapchain texture's VkImage, or none (stage A's colour) while nothing has written it
int vk_present_image(const tex_t *t, VkImage *img, uint32_t *w, uint32_t *h, int *srgb) {
    if (!t || !VK.ok) return 0;
    if (g_deferred) flush_from(FL_PRESENT);   // the frame's batched work goes to the GPU before the present that shows it
    pthread_mutex_lock(&g_tex_mu); vtex *v = tex_find(t->serial); int ok = v && v->img && (v->bad & 2);
    if (ok) { VkFormat f = tex_view_fmt(v, t); *img = v->img; *w = v->w; *h = v->h; *srgb = f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
    pthread_mutex_unlock(&g_tex_mu);
    return ok;
}
