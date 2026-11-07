// Test harness: the Linux host's Vulkan replay (gfx_vk_replay.c + gfx_vk_draw.inc + gfx_vk_shader.c) on any Vulkan device (lavapipe works) with the validation layer, fed a synthetic NVN
// command stream: a clear, a textured draw into the upper half of a 64x64 target (upper-left origin, as Mario Tennis Fever), a compute dispatch writing a storage buffer. Run: tests/vk/run.sh
#include "dm.h"
#include "../../../gfx/nvn/nvn_obj.h"
#include "../../../gfx/nvn/nvn_mtl.h"
#include "../../gfx_vk.h"
#include <dlfcn.h>
#define VK_DEF(f) PFN_##f f##_;
VKI_FUNCS(VK_DEF) VKD_FUNCS(VK_DEF)
vk_core VK;
static int g_val_errors;
uint64_t guest_ns(void) { return 0; }
// the dirty-page tracker (nvn_dirty.c) is not part of the test: nothing is tracked
int nvn_dirty_tracked(const void *a, size_t n) { return 0; } uint64_t nvn_dirty_now(void) { return 0; } void nvn_dirty_protect(const void *a, size_t n) {}
const void *nvn_dirty_span(const void *a, size_t n) { return NULL; } int nvn_dirty_since_span(const void *s, const void *a, size_t n, uint64_t st) { return 0; }
void dm_guest_write_prepare(void *a, size_t n) {}
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]) { if (st >= p->nstage) return 0; memcpy(key, p->sd[st].control, 8); return 1; }
int vkw_wanted(void) { return 1; }
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want) { for (uint32_t i = 0; i < VK.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (VK.mem.memoryTypes[i].propertyFlags & want) == want) return i; return ~0u; }
static VKAPI_ATTR VkBool32 VKAPI_CALL cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT t, const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
    if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) g_val_errors++; fprintf(stderr, "[VAL%s] %s\n\n", sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? " ERROR" : "", d->pMessage); }
    return 0;
}
int vkw_init(void) {
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW);
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance ci_ = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"}; const char *ext[] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "t", 1, "t", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 1, layers, 1, ext};
    if (ci_(&ici, NULL, &VK.inst)) { LOG("no instance"); return 0; }
#define LI(f) f##_ = (PFN_##f)gipa(VK.inst, #f);
    VKI_FUNCS(LI)
    PFN_vkCreateDebugUtilsMessengerEXT cm = (PFN_vkCreateDebugUtilsMessengerEXT)gipa(VK.inst, "vkCreateDebugUtilsMessengerEXT");
    VkDebugUtilsMessengerCreateInfoEXT mi = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0, 0x1100, 0x7, cb, NULL}; VkDebugUtilsMessengerEXT m; cm(VK.inst, &mi, NULL, &m);
    uint32_t n = 4; VkPhysicalDevice pd[4]; vkEnumeratePhysicalDevices_(VK.inst, &n, pd); VK.phys = pd[0];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties_(VK.phys, &pp); VK.limits = pp.limits; LOG("device %s", pp.deviceName);
    static VkPhysicalDeviceDepthClipControlFeaturesEXT fd = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    static VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fd};
    static VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    static VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    static VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    ((PFN_vkGetPhysicalDeviceFeatures2)gipa(VK.inst, "vkGetPhysicalDeviceFeatures2"))(VK.phys, &f2);
    f2.features.robustBufferAccess = 0; f2.features.shaderFloat64 = 0;   // as the Thor has none
    float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &pr};
    const char *dext[] = {VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &q, 0, NULL, 1, dext, NULL};
    if (vkCreateDevice_(VK.phys, &dci, NULL, &VK.dev)) { LOG("no device"); return 0; }
#define LD(f) f##_ = (PFN_##f)vkGetDeviceProcAddr_(VK.dev, #f); if (!f##_) LOG("missing " #f);
    VKD_FUNCS(LD)
    vkGetDeviceQueue_(VK.dev, 0, 0, &VK.q); vkGetPhysicalDeviceMemoryProperties_(VK.phys, &VK.mem); pthread_mutex_init(&VK.qmu, NULL);
    VK.depth_clip_control = fd.depthClipControl; VK.f_depth_clamp = 1; VK.ok = 1; return 1;
}
void vkw_present(uint64_t n, VkImage img, uint32_t w, uint32_t h, const int crop[4]) {}
int vkw_active(void) { return VK.ok; }

static uint32_t *sbuf_p;
static uint8_t cmd[1 << 16]; static size_t cn;
static void *rec(uint32_t op, uint32_t payload) { uint32_t sz = (8 + payload + 7) & ~7u; op_hdr *h = (op_hdr *)(cmd + cn); h->op = op; h->size = sz; memset(cmd + cn + 8, 0, sz - 8); cn += sz; return cmd + cn - sz + 8; }
static uint64_t be_key(const char *hex, uint8_t out[8]) { uint64_t k = strtoull(hex, NULL, 16); for (int i = 0; i < 8; i++) out[i] = (uint8_t)(k >> (56 - 8 * i)); return k; }

int main(int argc, char **argv) {
    setenv("DM_SHADERS", argc > 3 ? argv[3] : "/tmp/spvs.dmsp", 1); mtl_set_origin_lower_left(0);   /* upper left, as Mario Tennis */
    const char *vsk = argc > 1 ? argv[1] : "83477ae89102633b", *fsk = argc > 2 ? argv[2] : "831a03da67ed2702";
    mtl_sub_begin();
    if (getenv("DM_VK_DUMP_AT")) mtl_sub_commit();   // the frame dump starts with the submission after this empty one
    // render target 64x64 RGBA8, and a sampled 64x64 RGBA8 texture with block-linear storage (all 0x80)
    static tex_t rt, tx; memset(&rt, 0, sizeof rt);
    rt.d.target = NVN_TEXTURE_TARGET_2D; rt.d.format = NVN_FORMAT_RGBA8; rt.d.width = rt.d.height = 64; rt.d.depth = 1; rt.d.levels = 1; rt.serial = 1;
    rt.d.flags = NVN_TEXTURE_FLAGS_LINEAR; rt.d.stride = 256; rt.mem = calloc(1, 256 * 64);   // a linear render target: drawn pixels are read back into its guest memory
    tx = rt; tx.d.flags = 0; tx.d.stride = 0; tx.serial = 2; nvn_tex_layout(&tx.d, &tx.L); tx.mem = malloc(tx.L.size); memset(tx.mem, 0x80, tx.L.size);
    mtl_reg_tex(5, &tx, NULL);
    static view_t sv; sv.set_swizzle = 1; sv.swz[0] = 0; sv.swz[1] = 0; sv.swz[2] = 1; sv.swz[3] = 2; mtl_reg_tex(6, &tx, &sv);   /* a view swizzle: 0, 0, 1, R */
    samp_par sp = {NVN_MIN_FILTER_LINEAR, NVN_MAG_FILTER_LINEAR, NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_REPEAT, 0, 0, 0, 0, 1000, 0, 1, {0}};
    mtl_reg_samp(3, &sp);
    static uint8_t keys[2][8]; be_key(vsk, keys[0]); be_key(fsk, keys[1]);
    static prog_t prog; prog.nstage = 2; prog.sd[0].control = keys[0]; prog.sd[1].control = keys[1]; prog.serial = 77;
    static uint8_t gk[8]; if (argc > 4) { be_key(argv[4], gk); prog.nstage = 3; prog.sd[2].control = gk; }   // a geometry stage between them
    static float ubo[4096]; for (int i = 0; i < 4096; i++) ubo[i] = (i % 4 == 3) ? 1.0f : 0.25f;
    static float verts[3][12] = {{-1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1}, {3, 0, 0, 1, 2, 0, 0, 0, 1, 1, 1, 1}, {-1, 2, 0, 1, 0, 2, 0, 0, 1, 1, 1, 1}};   /* covers NDC y in [0, 1]: the upper half when +1 is the first row */
    p_set_rt *r = rec(OP_SET_RT, sizeof(p_set_rt)); r->n = 1; r->color[0] = &rt;
    p_clear_color *cc = rec(OP_CLEAR_COLOR, sizeof *cc); cc->index = 0; cc->mask = 0xf; cc->rgba[2] = 1; cc->rgba[3] = 1;
    p_bind_program *bp = rec(OP_BIND_PROGRAM, sizeof *bp); bp->prog = &prog; bp->stages = 0x3;
    p_bind_vattr *va = rec(OP_BIND_VATTR, sizeof *va); va->n = 3; for (int i = 0; i < 3; i++) { va_st a = {NVN_FORMAT_RGBA32F, 16u * i, 0}; memcpy(&va->st[i], &a, 4); }
    p_bind_vstream *vs = rec(OP_BIND_VSTREAM, sizeof *vs); vs->n = 1; vs->st[0].stride = 48;
    p_bind_vbuf1 *vb = rec(OP_BIND_VBUF1, sizeof *vb); vb->index = 0; vb->size = sizeof verts; vb->addr = (uint64_t)(uintptr_t)verts;
    for (int st = 0; st < 2; st++) for (int i = 0; i < 14; i++) { p_bind_buf *u = rec(OP_BIND_UBO, sizeof *u); u->stage = st; u->index = i; u->addr = (uint64_t)(uintptr_t)ubo; u->size = sizeof ubo; }
    p_bind_tex1 *bt = rec(OP_BIND_TEX1, 16); bt->stage = 1 | BIND_TEX1_NOBASE; bt->slot = 0; bt->h = 5 | (3u << 20);
    p_viewports *vp = rec(OP_VIEWPORTS, sizeof *vp); vp->n = 1; vp->v[0][2] = 64; vp->v[0][3] = 64;
    p_draw *d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_TRIANGLES; d->count = 3; d->instances = 1;
    d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_TRIANGLES; d->count = 3; d->instances = 1;   // the second draw reuses the pipeline and ring ranges
    // a quad over the lower-left quarter (NDC x, y in [-1, 0]): Vulkan has no quads, the replay makes two triangles of it
    static float quad[4][12] = {{-1, -1, 0, 1, 0, 0, 0, 0}, {0, -1, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 1, 0, 0}, {-1, 0, 0, 1, 0, 1, 0, 0}};
    vb = rec(OP_BIND_VBUF1, sizeof *vb); vb->index = 0; vb->size = sizeof quad; vb->addr = (uint64_t)(uintptr_t)quad;
    d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_QUADS; d->count = 4; d->instances = 1;
    // the alpha test (alpha 0.5 here): GREATER 0.6 over columns 32-47 of the lower half discards, LESS 0.6 over columns 48-63 draws
    static float q2[4][12] = {{0, -1, 0, 1, 0, 0, 0, 0}, {0.5f, -1, 0, 1, 1, 0, 0, 0}, {0.5f, 0, 0, 1, 1, 1, 0, 0}, {0, 0, 0, 1, 0, 1, 0, 0}};
    static float q3[4][12] = {{0.5f, -1, 0, 1, 0, 0, 0, 0}, {1, -1, 0, 1, 1, 0, 0, 0}, {1, 0, 0, 1, 1, 1, 0, 0}, {0.5f, 0, 0, 1, 0, 1, 0, 0}};
    { p_bind_state *bs = rec(OP_BIND_STATE, sizeof *bs); bs->kind = 1; color_st cs = {0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_GREATER}; memcpy(bs->bytes, &cs, sizeof cs);
      p_alpha_ref *ar = rec(OP_ALPHA_REF, sizeof *ar); ar->ref = 0.6f;
      vb = rec(OP_BIND_VBUF1, sizeof *vb); vb->index = 0; vb->size = sizeof q2; vb->addr = (uint64_t)(uintptr_t)q2;
      d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_QUADS; d->count = 4; d->instances = 1;
      bs = rec(OP_BIND_STATE, sizeof *bs); bs->kind = 1; cs = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_LESS}; memcpy(bs->bytes, &cs, sizeof cs);
      vb = rec(OP_BIND_VBUF1, sizeof *vb); vb->index = 0; vb->size = sizeof q3; vb->addr = (uint64_t)(uintptr_t)q3;
      d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_QUADS; d->count = 4; d->instances = 1;
      bs = rec(OP_BIND_STATE, sizeof *bs); bs->kind = 1; cs = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; memcpy(bs->bytes, &cs, sizeof cs); }
    { static float q4[4][12] = {{0.75f, -1, 0, 1, 0, 0, 0, 0}, {1, -1, 0, 1, 1, 0, 0, 0}, {1, -0.75f, 0, 1, 1, 1, 0, 0}, {0.75f, -0.75f, 0, 1, 0, 1, 0, 0}};
      { uint8_t *up = rec(OP_UPDATE_UB, 16 + 16); uint64_t dst = (uint64_t)(uintptr_t)ubo, nn = 16; float half[4] = {0.5f, 0.5f, 0.5f, 1}; memcpy(up, &dst, 8); memcpy(up + 8, &nn, 8); memcpy(up + 16, half, 16); }   /* the uniform buffer changed between draws */
      p_bind_tex1 *b6 = rec(OP_BIND_TEX1, 16); b6->stage = 1 | BIND_TEX1_NOBASE; b6->slot = 0; b6->h = 6 | (3u << 20);
      vb = rec(OP_BIND_VBUF1, sizeof *vb); vb->index = 0; vb->size = sizeof q4; vb->addr = (uint64_t)(uintptr_t)q4;
      d = rec(OP_DRAW, sizeof *d); d->prim = NVN_DRAW_PRIMITIVE_QUADS; d->count = 4; d->instances = 1; }
    // a compute dispatch writing a storage buffer
    static uint32_t sbuf[16]; sbuf_p = sbuf; static uint8_t ck[8]; be_key("3333333333333333", ck);
    static prog_t cprog; cprog.nstage = 1; cprog.sd[0].control = ck; cprog.serial = 78;
    p_bind_program *cbp = rec(OP_BIND_PROGRAM, sizeof *cbp); cbp->prog = &cprog; cbp->stages = NVN_SHADER_STAGE_COMPUTE_BIT;
    p_bind_buf *sb = rec(OP_BIND_SSBO, sizeof *sb); sb->stage = 5; sb->index = 0; sb->addr = (uint64_t)(uintptr_t)sbuf; sb->size = sizeof sbuf;
    p_dispatch *dp = rec(OP_DISPATCH, sizeof *dp); dp->x = 2; dp->y = 1; dp->z = 1;
    rec(OP_END, 0);
    mtl_exec(cmd);
    mtl_sub_commit();
    // read the target back
    static uint32_t px[64 * 64]; cn = 0;
    // the first submission may still be in flight: a buffer copy on the CPU from the storage buffer it writes back waits for it first
    static uint32_t sbuf2[16]; { uint8_t *cp = rec(OP_COPY_B2B, 24); uint64_t s_ = (uint64_t)(uintptr_t)sbuf, d_ = (uint64_t)(uintptr_t)sbuf2, n_ = sizeof sbuf; memcpy(cp, &s_, 8); memcpy(cp + 8, &d_, 8); memcpy(cp + 16, &n_, 8); }
    p_copy_t2t *c = rec(OP_COPY_T2B, sizeof *c); c->src = (const tex_t *)(void *)px; c->dst = &rt; c->dr.width = 64; c->dr.height = 64; c->dr.depth = 1;
    rec(OP_END, 0);
    mtl_sub_begin(); mtl_exec(cmd); uint64_t f2 = mtl_sub_commit();
    LOG("buffer copy after the write-back in flight: %u %u ... %u (7 8 ... 14 expected)", sbuf2[0], sbuf2[1], sbuf2[7]);
    LOG("fence wait: %d (1 expected), completed %d", mtl_wait(f2, 2000000000ull), mtl_wait(f2, 0));
    LOG("view swizzle: row 63 col 63 %08x (80800000 expected: b = one x 0.5 after the uniform update, a = r x 1)", px[63 * 64 + 63]);
    LOG("alpha test: row 50 col 40 %08x (discarded: ffff0000), row 50 col 56 %08x (drawn: 80202020)", px[50 * 64 + 40], px[50 * 64 + 56]);
    LOG("pixels: row 0 %08x, row 20 %08x, row 40 %08x, row 63 %08x; quad: row 50 col 5 %08x, row 50 col 50 %08x", px[0], px[20 * 64 + 5], px[40 * 64 + 5], px[63 * 64 + 5], px[50 * 64 + 5], px[50 * 64 + 50]);
    { const uint32_t *g = (const uint32_t *)rt.mem; LOG("linear target's guest memory: row 0 %08x, row 50 col 5 %08x, row 50 col 50 %08x", g[0], g[50 * 64 + 5], g[50 * 64 + 50]); }
    LOG("storage buffer after the submission: %u %u %u %u ... %u", sbuf_p[0], sbuf_p[1], sbuf_p[2], sbuf_p[3], sbuf_p[7]);
    // an alias: an sRGB texture over the linear target's storage reads what was drawn into the UNORM one (one image)
    { static tex_t al; al = rt; al.serial = 9; al.d.format = NVN_FORMAT_RGBA8_SRGB; static uint32_t ap[64 * 64]; cn = 0;
      p_copy_t2t *c2 = rec(OP_COPY_T2B, sizeof *c2); c2->src = (const tex_t *)(void *)ap; c2->dst = &al; c2->dr.width = 64; c2->dr.height = 64; c2->dr.depth = 1;
      rec(OP_END, 0); mtl_sub_begin(); mtl_exec(cmd); mtl_sub_commit(); mtl_finish();
      LOG("alias: row 50 col 50 %08x (as the target: %08x)", ap[50 * 64 + 50], px[50 * 64 + 50]); }
    // a texture of another shape made later at the same storage does not hide the target from a later alias
    { static tex_t sm, a2; sm = rt; sm.serial = 10; sm.d.width = sm.d.height = 32; a2 = rt; a2.serial = 11; a2.d.format = NVN_FORMAT_RGBA8_SRGB; static uint32_t sp[32 * 32], bp[64 * 64]; cn = 0;
      p_copy_t2t *c3 = rec(OP_COPY_T2B, sizeof *c3); c3->src = (const tex_t *)(void *)sp; c3->dst = &sm; c3->dr.width = 32; c3->dr.height = 32; c3->dr.depth = 1;
      p_copy_t2t *c4 = rec(OP_COPY_T2B, sizeof *c4); c4->src = (const tex_t *)(void *)bp; c4->dst = &a2; c4->dr.width = 64; c4->dr.height = 64; c4->dr.depth = 1;
      rec(OP_END, 0); mtl_sub_begin(); mtl_exec(cmd); mtl_sub_commit(); mtl_finish();
      LOG("alias past another texture at the same storage: row 50 col 50 %08x (as the target: %08x)", bp[50 * 64 + 50], px[50 * 64 + 50]); }
    // batching: a clear into a texture is kept back (nothing in it lands in guest memory), the read-back after it takes both to the GPU in one submission
    { static tex_t bt; bt = tx; bt.serial = 12; bt.mem = calloc(1, tx.L.size); static uint32_t bpx[64 * 64]; cn = 0;
      p_set_rt *r2 = rec(OP_SET_RT, sizeof(p_set_rt)); r2->n = 1; r2->color[0] = &bt;
      p_clear_color *c5 = rec(OP_CLEAR_COLOR, sizeof *c5); c5->index = 0; c5->mask = 0xf; c5->rgba[0] = 1; c5->rgba[3] = 1;
      rec(OP_END, 0); mtl_sub_begin(); mtl_exec(cmd); mtl_sub_commit(); cn = 0;
      p_copy_t2t *c6 = rec(OP_COPY_T2B, sizeof *c6); c6->src = (const tex_t *)(void *)bpx; c6->dst = &bt; c6->dr.width = 64; c6->dr.height = 64; c6->dr.depth = 1;
      rec(OP_END, 0); mtl_sub_begin(); mtl_exec(cmd); mtl_sub_commit(); mtl_finish();
      LOG("batched clear then read-back: %08x (ff0000ff expected)", bpx[5]); }
    LOG("validation errors: %d", g_val_errors);
    return g_val_errors ? 1 : 0;
}
