// Stage C of the Vulkan graphics layer (dm/linux/GRAPHICS.md): draws and compute dispatches. Included by gfx_vk_replay.c (it uses the replay's command buffer, textures and barriers).
//
// First version, built for correctness over speed:
//   - buffers: guest memory is not visible to the GPU (Turnip has no VK_EXT_external_memory_host), so every range a draw reads (uniform, storage, vertex and index buffers, the
//     shader's own constants) is copied into an upload ring at the draw, once per submission for each (address, size); storage buffers a shader writes are copied back into
//     guest memory when the submission completes (a later draw of the same submission that binds the same range reads the ring's copy)
//   - textures: one VkImage per texture serial (gfx_vk_replay.c); a texture the GPU has not written gets its guest content (block-linear, nvn_deswizzle) at its first use
//   - pipelines: Vulkan 1.3 dynamic rendering, one pipeline per (program, render target formats, vertex layout, blend / depth / raster state), cached for the run
//   - descriptors: one set per stage per draw from per-submission pools
//   - coordinates: NVN's upper-left window origin puts NDC y = +1 on a target's first row, Vulkan puts -1 there, so upper-left draws get a viewport of negative height;
//     NVN's GL depth mode (near = -w) clips z in [-1, 1] (VK_EXT_depth_clip_control)
// Not yet: transform feedback, tessellation and geometry stages, conditional rendering, alpha test, logic ops, quads, texture buffers, bindless handles in uniform buffers
// (those draws are skipped or drawn without them; the summary line counts them).

// ---------------------------------------------------------------- the upload ring: one host-visible buffer in three parts, one per submission slot: a submission fills its part
// from the start while the GPU reads the others'
#define RING_SIZE (192ull << 20)
#define RING_ALIGN 256
#define RING_PART ((RING_SIZE / 3) & ~(uint64_t)(RING_ALIGN - 1))   // one submission slot's part
static struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; uint64_t used, base; int full_warned; } RG;   // base: the start of the recording slot's half
typedef struct { uint64_t addr, size, off; } ring_hit;
#define RH_CAP 8192
static ring_hit g_rh[RH_CAP]; static int g_nrh_used; static uint16_t g_rh_idx[RH_CAP];   // (address, size) → ring offset for this submission (open addressing, cleared at the submission's end)
static uint64_t g_ring_bytes, g_ring_full;
static int ring_init(void) {
    if (RG.b) return 1;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, RING_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                              VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer_(VK.dev, &bci, NULL, &RG.b)) { LOG("[RED] [vk] the upload ring buffer was refused"); return 0; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements_(VK.dev, RG.b, &mr);
    uint32_t mt = vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == ~0u) mt = vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt};
    if (mt == ~0u || vkAllocateMemory_(VK.dev, &ai, NULL, &RG.m)) { LOG("[RED] [vk] the upload ring's memory (%llu MB) was refused", (unsigned long long)(RING_SIZE >> 20)); return 0; }
    vkBindBufferMemory_(VK.dev, RG.b, RG.m, 0); vkMapMemory_(VK.dev, RG.m, 0, VK_WHOLE_SIZE, 0, (void **)&RG.p);
    LOG("[vk] upload ring: %llu MB", (unsigned long long)(RING_SIZE >> 20));
    return 1;
}
static uint64_t ring_alloc(uint64_t n) {
    uint64_t off = (RG.used + RING_ALIGN - 1) & ~(uint64_t)(RING_ALIGN - 1);
    uint64_t n16 = (n + 15) & ~15ull;   // reserved in whole 16-byte rows: a binding's range may round up past the copied bytes
    if (off + n16 > RG.base + RING_PART) { g_ring_full++; if (!RG.full_warned++) LOG("[RED] [vk] the upload ring is full within one submission: draws past it are skipped"); return ~0ull; }
    RG.used = off + n16; g_ring_bytes += n; return off;
}
static int fwd_try(uint64_t addr, uint64_t size, uint64_t ring_off);   // gfx_vk_replay.c
// The guest's bytes [addr, addr + size) in the ring (the same range again in this submission: the same copy)
static uint64_t ring_put(uint64_t addr, uint64_t size) {
    if (!size) size = 16;
    uint32_t i = (uint32_t)((addr ^ size * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull >> 51) & (RH_CAP - 1);
    for (int k = 0; k < 32; k++, i = (i + 1) & (RH_CAP - 1)) {
        if (g_rh[i].addr == addr && g_rh[i].size == size) return g_rh[i].off;
        if (!g_rh[i].addr) break;
    }
    uint64_t off = ring_alloc(size); if (off == ~0ull) return off;
    int fwd = addr && fwd_try(addr, size, off);   // a storage buffer result of a submission in flight: copied on the GPU (else waited for)
    if (addr && !fwd) pend_hazard(addr, size);
    if (addr) memcpy(RG.p + off, (const void *)(uintptr_t)addr, size); else memset(RG.p + off, 0, size);
    if (addr && !g_rh[i].addr && g_nrh_used < RH_CAP) { g_rh[i] = (ring_hit){addr, size, off}; g_rh_idx[g_nrh_used++] = (uint16_t)i; }
    return off;
}
static double ring_mb_per_sub(uint64_t n) { static uint64_t last; double r = (g_ring_bytes - last) / 1048576.0 / (n ? n : 1); last = g_ring_bytes; return r; }
// Guest bytes the command stream changes as it goes (nvnCommandBufferUpdateUniformBuffer, buffer copies and clears): copies of them made earlier in this submission
// are not reused, so the draws after the change see the new bytes. Particle effects update one uniform buffer between their emitters' draws: every emitter drew
// with the first one's constants, and their alpha curves and colours came out as solid shapes
static uint64_t g_ring_forgot;
void ring_forget(uint64_t addr, uint64_t n) {
    if (!g_nrh_used || !n) return;
    for (int k = 0; k < g_nrh_used; k++) {
        ring_hit *h = &g_rh[g_rh_idx[k]];
        if (h->addr && h->addr < addr + n && addr < h->addr + h->size) { h->addr = 0; h->size = 0; g_ring_forgot++; }   // later lookups past this slot miss and copy again
    }
}
// Storage buffers shaders wrote in this submission: copied back into guest memory when it completes, so the game (and the next submission's copies) see the results
// Only the bytes the GPU changed go back (orig: the copy as recorded): the submission completes while the game runs on, and the rest of the buffer may hold what the CPU
// wrote since (the next frame's data); writing the whole range back put the old bytes over it
typedef struct { uint64_t addr, size, off; uint8_t *orig; int64_t fwo; uint32_t fw_epoch; } wback;   // fwo: its copy in the forwarding buffer (-1: none)
static wback *g_wb; static int g_nwb, g_cwb; static uint64_t g_wb_bytes, g_wb_changed;
static void wb_add(uint64_t addr, uint64_t size, uint64_t off) {
    for (int i = 0; i < g_nwb; i++) if (g_wb[i].addr == addr && g_wb[i].size == size) return;
    if (g_nwb == g_cwb) { g_cwb = g_cwb ? 2 * g_cwb : 64; g_wb = realloc(g_wb, g_cwb * sizeof *g_wb); }
    uint8_t *o = malloc(size); if (o) memcpy(o, RG.p + off, size);
    g_wb[g_nwb++] = (wback){addr, size, off, o, -1, 0};
}
// Storage buffer results kept on the GPU: at the end of a submission each range a shader wrote is also copied into the forwarding buffer, so a later submission that reads
// those bytes before the GPU is done (and before they are back in guest memory) copies them on the GPU instead of waiting. Mario Tennis Fever's compute writes a small
// buffer each frame that the next frame reads: the replay waited for the whole frame to finish, every frame. DM_VK_FORWARD=0 waits
#define FW_SIZE (16ull << 20)
static struct { VkBuffer b; VkDeviceMemory m; uint64_t used; uint32_t epoch; int failed; } FW;
typedef struct { uint64_t addr, size, off; } fwent;
static fwent g_fw[1024]; static int g_nfw; static uint64_t g_fwd_n, g_fwd_miss, g_fwd_post;
static int fw_init(void) {
    if (FW.b) return 1;
    if (FW.failed) return 0;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, FW_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer_(VK.dev, &bci, NULL, &FW.b)) { FW.failed = 1; return 0; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements_(VK.dev, FW.b, &mr);
    uint32_t mt = vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, mt};
    if (mt == ~0u || vkAllocateMemory_(VK.dev, &ai, NULL, &FW.m)) { vkDestroyBuffer_(VK.dev, FW.b, NULL); FW.b = VK_NULL_HANDLE; FW.failed = 1; return 0; }
    vkBindBufferMemory_(VK.dev, FW.b, FW.m, 0);
    return 1;
}
static int64_t fw_slot(uint64_t addr, uint64_t size) {   // the forwarding buffer's place for this range (a new epoch when it runs out: older places are not trusted)
    for (int i = 0; i < g_nfw; i++) if (g_fw[i].addr == addr && g_fw[i].size == size) return (int64_t)g_fw[i].off;
    uint64_t n = (size + 255) & ~255ull;
    if (n > FW_SIZE) return -1;
    if (g_nfw == 1024 || FW.used + n > FW_SIZE) { g_nfw = 0; FW.used = 0; FW.epoch++; }
    g_fw[g_nfw++] = (fwent){addr, size, FW.used}; FW.used += n;
    return (int64_t)(FW.used - n);
}
static void fw_copy_out(void) {   // at the submission's end (no pass open): its write-backs into the forwarding buffer
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_FORWARD"); on = !(e && *e == '0'); }
    int any = g_nwb; for (int i = 0; i < g_nstage && !any; i++) any = g_stage[i].orig != NULL;
    if (!on || !any || !fw_init()) return;
    full_barrier();
    for (int i = 0; i < g_nwb; i++) {
        wback *w = &g_wb[i]; int64_t o = fw_slot(w->addr, w->size); if (o < 0) continue;
        VkBufferCopy bc = {w->off, (uint64_t)o, w->size}; vkCmdCopyBuffer_(g_cb, RG.b, FW.b, 1, &bc);
        w->fwo = o; w->fw_epoch = FW.epoch;
    }
    for (int i = 0; i < g_nstage; i++) {   // packed texture-to-buffer copies (their staging buffer holds the bytes as the guest will)
        stage_t *st = &g_stage[i]; if (!st->rb_dst || !st->orig) continue;
        uint64_t n = (st->rb_rows - 1) * st->rb_row + st->rb_n; int64_t o = fw_slot((uint64_t)(uintptr_t)st->rb_dst, n); if (o < 0) continue;
        VkBufferCopy bc = {0, (uint64_t)o, n}; vkCmdCopyBuffer_(g_cb, st->b, FW.b, 1, &bc);
        st->fwo = o; st->fw_epoch = FW.epoch;
    }
}
static void wb_land(const wback *w) {   // on the completion thread
    uint8_t *g = (uint8_t *)(uintptr_t)w->addr; const uint8_t *r = RG.p + w->off;
    g_wb_bytes += w->size;
    if (!w->orig) { dm_guest_write_prepare(g, w->size); memcpy(g, r, w->size); return; }
    uint64_t i = 0;
    while (i < w->size) {
        if (w->size - i >= 8) { uint64_t a, b; memcpy(&a, r + i, 8); memcpy(&b, w->orig + i, 8); if (a == b) { i += 8; continue; } }
        else if (r[i] == w->orig[i]) { i++; continue; }
        while (r[i] == w->orig[i]) i++;   // the first changed byte (one is in these 8, or it is this one)
        uint64_t j = i; while (j < w->size && r[j] != w->orig[j]) j++;   // the run of changed bytes (an unchanged byte may be one the CPU wrote since: left alone)
        dm_guest_write_prepare(g + i, j - i); memcpy(g + i, r + i, j - i); g_wb_changed += j - i;
        i = j;
    }
}
static uint64_t ring_bytes(const void *p, uint64_t n) { uint64_t off = ring_alloc(n); if (off != ~0ull) memcpy(RG.p + off, p, n); return off; }

// ---------------------------------------------------------------- texture, sampler and image registrations (the game's descriptor pools: hle_nvn.c)
typedef struct { tex_t t; view_t v; int has_v; } treg;
static treg **g_treg; static samp_par **g_sreg; static int g_ntreg, g_nsreg;
static pthread_mutex_t g_reg_mu = PTHREAD_MUTEX_INITIALIZER;
static void reg_put(void ***tab, int *n, int id, void *e) {
    if (id < 0 || id >= (1 << 22)) { free(e); return; }
    pthread_mutex_lock(&g_reg_mu);
    if (id >= *n) { int m = *n ? *n : 1024; while (m <= id) m *= 2; *tab = realloc(*tab, m * sizeof(void *)); memset(*tab + *n, 0, (m - *n) * sizeof(void *)); *n = m; }
    void *old = (*tab)[id]; (*tab)[id] = e;
    pthread_mutex_unlock(&g_reg_mu);
    free(old);   // a registration replaced while a submission reads the old one is the game's race (NVN has it too)
}
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) { treg *e = calloc(1, sizeof *e); if (t) e->t = *t; if (v) { e->v = *v; e->has_v = 1; } reg_put((void ***)&g_treg, &g_ntreg, id, e); }
// Images share the texture pool's index space (an image handle is a pool index: hle_nvn.c nvnDeviceGetImageHandle), so RegisterImage writes the same table as RegisterTexture
void mtl_reg_image(int id, const tex_t *t, const view_t *v) { treg *e = calloc(1, sizeof *e); if (t) { e->t = *t; e->t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; } if (v) { e->v = *v; e->has_v = 1; } reg_put((void ***)&g_treg, &g_ntreg, id, e); }
void mtl_reg_samp(int id, const samp_par *p) { samp_par *e = malloc(sizeof *e); *e = *p; reg_put((void ***)&g_sreg, &g_nsreg, id, e); }
void mtl_set_tex_pool_n(int n) { (void)n; }
static const treg *treg_get(int id) { pthread_mutex_lock(&g_reg_mu); const treg *r = id > 0 && id < g_ntreg ? g_treg[id] : NULL; pthread_mutex_unlock(&g_reg_mu); return r && r->t.serial ? r : NULL; }
static const treg *ireg_get(int id) { return treg_get(id); }
static const samp_par *sreg_get(int id) { pthread_mutex_lock(&g_reg_mu); const samp_par *r = id >= 0 && id < g_nsreg ? g_sreg[id] : NULL; pthread_mutex_unlock(&g_reg_mu); return r; }

// Window origin and depth mode (hle_nvn.c)
static _Atomic int g_origin_ll = 1, g_depth_zero;
static int g_queue_origin_ll;
void mtl_set_origin_lower_left(int on) { atomic_store(&g_origin_ll, on != 0); }
void mtl_set_queue_origin_lower_left(int on) { g_queue_origin_ll = on != 0; }
void mtl_set_depth_zero(int on) { atomic_store(&g_depth_zero, on != 0); }

// The CPU wrote a texture's storage: unless the GPU owns it, it is uploaded again at its next use
void mtl_tex_cpu_written(const tex_t *t) {
    if (!t || !t->serial) return;
    pthread_mutex_lock(&g_tex_mu); vtex *v = tex_find(t->serial); if (v && !v->gpu) v->uploaded = 0; pthread_mutex_unlock(&g_tex_mu);
}
// The CPU wrote a registered texture's pages after its first use (hle_nvn.c nvn_ft_poll, profile key tex_dirty_upload): uploaded again from its storage, unless the GPU
// has written the texture (then its content is the GPU's: left to it until the CPU rewrites it)
int mtl_tex_cpu_rewritten(const tex_t *t) {
    if (!t || !t->serial) return MTL_REWRITE_UPLOADED;
    pthread_mutex_lock(&g_tex_mu); vtex *v = tex_find(t->serial); int gpu = v && v->gpu && !v->linear; if (v && !gpu) v->uploaded = 0; pthread_mutex_unlock(&g_tex_mu);
    return gpu ? MTL_REWRITE_GPU_NOW : MTL_REWRITE_UPLOADED;
}
int mtl_tex_gpu_owned(uint64_t serial) { pthread_mutex_lock(&g_tex_mu); vtex *v = tex_find(serial); int r = v && v->gpu; pthread_mutex_unlock(&g_tex_mu); return r; }

// ---------------------------------------------------------------- texture contents from guest memory
static uint64_t g_up_tex, g_up_bytes, g_up_fail, g_reup;
// CPU writes after an upload: the texture's guest pages are write-protected (nvn_dirty.c, the pools are tracked ranges), so the game's next write there faults once and is
// recorded; prep_tex sees it and the texture is uploaded again. NVN is not told about such writes (a font's glyph sheet, a movie's frames)
static uint64_t g_untracked;
static void track(vtex *v) {
    if (!v->gmem || !v->gsize) return;
    if (!nvn_dirty_tracked(v->gmem, v->gsize)) { v->tracked = 0; g_untracked++; return; }
    v->stamp = nvn_dirty_now(); nvn_dirty_protect(v->gmem, v->gsize); v->span = nvn_dirty_span(v->gmem, v->gsize); v->tracked = 1;
}
static int cpu_wrote(vtex *v) { return v->tracked && nvn_dirty_since_span(v->span, v->gmem, v->gsize, v->stamp); }
// DM_VK_TEXLOG=1: one line per texture upload (and every failure is logged, the first 20)
static int texlog(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_TEXLOG"); on = e && *e == '1'; } return on; }
#define UPFAIL(why) do { g_up_fail++; static int nf_; if (nf_++ < 20 || texlog()) LOG("[RED] [vk] texture upload failed (%s): serial %llu format %#x target %d %ux%ux%u levels %u flags %#x samples %d", why, \
    (unsigned long long)t->serial, t->d.format, t->d.target, t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.flags, t->d.samples); } while (0)
static void tex_fill(vtex *v, const tex_t *t) {
    if (v && !v->gpu && !v->uploaded && t && !t->mem && texlog()) { static int n; if (n++ < 50) LOG("[vk] texture serial %llu format %#x %ux%u has no storage pointer: not uploaded", (unsigned long long)t->serial, t->d.format, t->d.width, t->d.height); }
    if (!v || (v->gpu && !v->linear) || v->uploaded || !t || !t->mem || v->samples > 1 || v->sc != 1.0f) return;   // a scaled render target: no guest content goes in
    v->uploaded = 1;
    // A texture in a sparse virtual pool (flags 0x200, vmode 2): the CPU never writes there (NVN gives virtual pools no CPU mapping; what the CPU writes goes through
    // the physical pool, and an aliased pool, vmode 1, reads that memory), and its unmapped pages have no memory behind them: reading them stopped the game with
    // SIGBUS when Mario Tennis Fever's court loaded. Its contents are only what the GPU draws: nothing to upload
    if (t->pool && (t->pool->flags & 0x200) && t->pool->vmode != 1) {
        static int n; if (n++ < 10) LOG("[vk] texture serial %llu (%ux%u) is in a sparse virtual pool: starts empty, not read from memory", (unsigned long long)t->serial, t->d.width, t->d.height);
        return;
    }
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) { UPFAIL("unknown format size"); return; }
    int linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;
    uint32_t layers = v->is3d ? 1 : v->layers, levels = v->levels;
    if (linear) levels = 1;
    // the whole texture in one staging buffer: level-major, every layer of a level after the other, block rows packed
    uint64_t total = 0;
    for (uint32_t l = 0; l < levels; l++) {
        uint32_t w = v->w >> l ? v->w >> l : 1, h = v->h >> l ? v->h >> l : 1, d = v->is3d ? (v->d >> l ? v->d >> l : 1) : 1;
        total += (uint64_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * d * layers;
    }
    if (t->pool && t->pool->mem && t->pool->size && (t->mem < t->pool->mem || (uint64_t)(t->mem - t->pool->mem) + total > t->pool->size)) {   // past its pool: logged
        static int n; if (n++ < 10) LOG("[YELLOW] [vk] texture serial %llu (%ux%u, %llu bytes) at offset %#llx runs past its pool (%#llx bytes, flags %#x, mode %d)", (unsigned long long)t->serial, t->d.width, t->d.height,
            (unsigned long long)total, (unsigned long long)(t->mem - t->pool->mem), (unsigned long long)t->pool->size, t->pool->flags, t->pool->vmode);
    }
    pend_hazard((uint64_t)(uintptr_t)t->mem, linear && t->d.stride ? (uint64_t)t->d.stride * ((v->h + bh - 1) / bh) : total);
    stage_t *s = stage_new(total); if (!s) { UPFAIL("no staging memory"); return; }
    VkBufferImageCopy bic[32]; uint32_t nb = 0; uint64_t off = 0;
    VkImageAspectFlags asp = fmt_aspects(v->fmt);
    if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { UPFAIL("packed depth-stencil"); return; }   // packed depth-stencil storage needs splitting into planes: not yet
    for (uint32_t l = 0; l < levels && nb < 32; l++) {
        uint32_t w = v->w >> l ? v->w >> l : 1, h = v->h >> l ? v->h >> l : 1, d = v->is3d ? (v->d >> l ? v->d >> l : 1) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh, lsz = pitch * rows * d;
        for (uint32_t a = 0; a < layers; a++) {
            uint8_t *dst = (uint8_t *)s->p + off + a * lsz;
            if (linear) { size_t st = t->d.stride ? t->d.stride : pitch; for (size_t r = 0; r < rows; r++) memcpy(dst + r * pitch, t->mem + r * st, pitch); }
            else if (nvn_deswizzle(&t->d, l, a, t->mem, dst, pitch)) { memset(dst, 0, lsz); UPFAIL("deswizzle"); }
        }
        bic[nb++] = (VkBufferImageCopy){off, 0, 0, {asp, l, 0, layers}, {0, 0, 0}, {w, h, d}};
        off += lsz * layers;
    }
    vkCmdCopyBufferToImage_(g_cb, s->b, v->img, VK_IMAGE_LAYOUT_GENERAL, nb, bic);
    full_barrier();
    g_up_tex++; g_up_bytes += total;
    if (texlog()) LOG("[vk] texture upload: serial %llu format %#x (Vulkan %d) target %d %ux%ux%u levels %u layers %u flags %#x%s swizzle %s%#x, %.1f KB, first word %08x",
                      (unsigned long long)t->serial, t->d.format, v->fmt, t->d.target, t->d.width, t->d.height, t->d.depth, levels, layers, t->d.flags, linear ? " (linear)" : "",
                      t->set_swz ? "set " : "", t->swz, total / 1024.0, *(const uint32_t *)s->p);
    track(v);
}

// ---------------------------------------------------------------- samplers
static const VkSamplerAddressMode k_wrap[8] = {VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE /* GL clamp */, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
    VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
typedef struct { samp_par p; VkSampler s; } scache;
#define SC_CAP 4096
static scache g_sc[SC_CAP]; static int g_nsc;
static VkSampler sampler_for(const samp_par *p) {
    static const samp_par dflt = {NVN_MIN_FILTER_LINEAR, NVN_MAG_FILTER_LINEAR, NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_REPEAT, 0, 0, 0, 0, 1000, 0, 1, {0}};
    if (!p) p = &dflt;
    uint32_t i = 0; for (size_t k = 0; k < sizeof *p; k++) i = i * 31 + ((const uint8_t *)p)[k];
    i &= SC_CAP - 1;
    for (int k = 0; k < SC_CAP; k++, i = (i + 1) & (SC_CAP - 1)) {
        if (g_sc[i].s && !memcmp(&g_sc[i].p, p, sizeof *p)) return g_sc[i].s;
        if (!g_sc[i].s) break;
    }
    int mip = p->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, p->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        (p->minf == NVN_MIN_FILTER_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        (p->minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        k_wrap[p->ws & 7], k_wrap[p->wt & 7], k_wrap[p->wr & 7], p->lod_bias, p->aniso > 1.0f, p->aniso > 1.0f ? (p->aniso < 16 ? p->aniso : 16) : 1.0f,
        p->cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, (VkCompareOp)(p->cmp_func ? p->cmp_func - 1 : 7), mip ? p->lod_min : 0, mip ? p->lod_max : 0.25f,
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
    if (p->border[3] >= 0.5f) ci.borderColor = p->border[0] >= 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    VkSamplerReductionModeCreateInfo red = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN :
                                           p->reduction == NVN_SAMPLER_REDUCTION_MAX ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    if (p->reduction) ci.pNext = &red;
    VkSampler s = VK_NULL_HANDLE;
    if (vkCreateSampler_(VK.dev, &ci, NULL, &s)) return VK_NULL_HANDLE;
    if (g_nsc < SC_CAP * 3 / 4) { g_sc[i].p = *p; g_sc[i].s = s; g_nsc++; }
    return s;
}

// ---------------------------------------------------------------- image views (per texture, by what the binding asks: view type, levels, layers, swizzle, aspect)
// The same-size format whose texels read as the other numeric kind (integers ↔ floats), for a view a shader takes of a texture with MUTABLE_FORMAT; 0 = none
static VkFormat other_kind(VkFormat f, int want_int) {
    // only 32-bit channels: their texels keep their bits through a view of the other kind (the translator's shader bit-casts what it reads); a normalized or
    // 16-bit float view would convert the values instead
    static const VkFormat pairs[][3] = {   // float, unsigned, signed
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT}, {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT},
        {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT}};
    for (size_t i = 0; i < sizeof pairs / sizeof *pairs; i++) {
        if (pairs[i][0] == f) return want_int ? pairs[i][1] : 0;
        if (pairs[i][1] == f || (pairs[i][2] && pairs[i][2] == f)) return want_int ? 0 : pairs[i][0];
    }
    return 0;
}
// A storage view's format: sRGB formats cannot be shader images, their UNORM twin can (the shader writes encoded values, as NVN's image view of an sRGB texture does)
static VkFormat storage_twin(VkFormat f) {
    switch (f) { case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM; case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32; default: return f; }
}
typedef struct vview { struct vview *next; VkImageViewType type; VkFormat fmt; uint32_t lv, nl, l0, nlay; VkComponentMapping sw; VkImageAspectFlags asp; int storage; VkImageView view; } vview;
static void vk_views_drop(vtex *v) { for (vview *w = v->views, *n; w; w = n) { n = w->next; vkDestroyImageView_(VK.dev, w->view, NULL); free(w); } v->views = NULL; }
static VkImageView view_get_fmt(vtex *v, VkFormat fmt, VkImageViewType type, uint32_t lv, uint32_t nl, uint32_t l0, uint32_t nlay, VkComponentMapping sw, VkImageAspectFlags asp, int storage);
static VkImageView view_get(vtex *v, VkImageViewType type, uint32_t lv, uint32_t nl, uint32_t l0, uint32_t nlay, VkComponentMapping sw, VkImageAspectFlags asp, int storage) {
    return view_get_fmt(v, v->fmt, type, lv, nl, l0, nlay, sw, asp, storage);
}
static VkImageView view_get_fmt(vtex *v, VkFormat fmt, VkImageViewType type, uint32_t lv, uint32_t nl, uint32_t l0, uint32_t nlay, VkComponentMapping sw, VkImageAspectFlags asp, int storage) {
    for (vview *w = v->views; w; w = w->next)
        if (w->type == type && w->fmt == fmt && w->lv == lv && w->nl == nl && w->l0 == l0 && w->nlay == nlay && !memcmp(&w->sw, &sw, sizeof sw) && w->asp == asp && w->storage == storage) return w->view;
    VkImageViewUsageCreateInfo use = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &use, 0, v->img, type, fmt, sw, {asp, lv, nl, l0, nlay}};
    if (storage == 2) { use.usage = fmt_depth(v->fmt) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }   // 2: an attachment view
    VkImageView iv = VK_NULL_HANDLE;
    if (vkCreateImageView_(VK.dev, &ci, NULL, &iv)) return VK_NULL_HANDLE;
    vview *w = calloc(1, sizeof *w); *w = (vview){v->views, type, fmt, lv, nl, l0, nlay, sw, asp, storage, iv}; v->views = w;
    return iv;
}
// The view a shader binding takes of a texture: its type from the shader's image (SPIR-V Dim, arrayed, multisampled), clamped to what the texture has
static VkImageView tex_view(vtex *v, const tex_t *t, const view_t *vw, int has_vw, uint8_t img, int storage) {
    uint32_t lv, nl, l0, nlay; view_range(v, vw, has_vw, &lv, &nl, &l0, &nlay);
    int dim = img & 15, arr = (img >> 4) & 1, ms = (img >> 5) & 1;
    if (ms != (v->samples > 1)) return VK_NULL_HANDLE;
    VkImageViewType type;
    switch (dim) {
    case 0: if (!v->is1d) return VK_NULL_HANDLE; type = arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D; break;
    case 1: case 4: if (v->is1d || v->is3d) return VK_NULL_HANDLE; type = arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; break;
    case 2: if (!v->is3d) return VK_NULL_HANDLE; type = VK_IMAGE_VIEW_TYPE_3D; l0 = 0; nlay = 1; break;
    case 3: if (!v->cube || v->layers < 6) return VK_NULL_HANDLE;   // only an image made cube-compatible (square, 6+ layers)
            type = arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE; if (nlay < 6) { l0 = 0; nlay = 6; } if (!arr) nlay = 6; else nlay -= nlay % 6; break;
    default: return VK_NULL_HANDLE;
    }
    if (!arr && dim != 3) nlay = 1;
    if (storage) nl = 1;
    // Swizzle: the texture's own (TextureBuilderSetSwizzle: 0 zero 1 one 2-5 R G B A), then the one the format mapping needs (nvn_fmt_to_mtl, the same encoding)
    uint32_t mf = 0; uint8_t fs[4] = {2, 3, 4, 5}; nvn_fmt_to_mtl(t->d.format, &mf, fs);
    uint8_t ts[4] = {2, 3, 4, 5};
    if (t->set_swz) for (int c = 0; c < 4; c++) ts[c] = (t->swz >> (3 * c)) & 7;
    if (has_vw && vw->set_swizzle) for (int c = 0; c < 4; c++) ts[c] = vw->swz[c];
    VkComponentSwizzle out[4];
    for (int c = 0; c < 4; c++) { uint8_t x = ts[c] >= 2 && ts[c] <= 5 ? fs[ts[c] - 2] : ts[c]; out[c] = x <= 5 ? (VkComponentSwizzle)(x + 1) : VK_COMPONENT_SWIZZLE_IDENTITY; }
    VkComponentMapping sw = {out[0], out[1], out[2], out[3]};
    if (storage) sw = (VkComponentMapping){0, 0, 0, 0};
    VkImageAspectFlags asp = fmt_aspects(v->fmt);
    if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) asp = (t->stencil || (has_vw && vw->set_dsmode && vw->dsmode)) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    // the view's format: the shader's numeric kind of texel (the translator types a texel it has no answer for as float), and a storage-capable twin for sRGB shader images
    VkFormat fmt = tex_view_fmt2(v, t, vw, has_vw);   // an alias's own format or the view's (sRGB or UNORM)
    if (v->mutable_fmt && !fmt_depth(fmt)) {
        int want_int = (img >> 7) & 1;
        if (want_int != fmt_int(fmt)) { VkFormat o = other_kind(fmt, want_int); if (o) fmt = o; }
        if (storage) fmt = storage_twin(fmt);
    }
    if (storage && v->storage == 2 && fmt == v->fmt) return VK_NULL_HANDLE;   // its own format cannot be a shader image, only a twin can
    return view_get_fmt(v, fmt, type, lv, nl, l0, nlay, sw, asp, storage);
}

// ---------------------------------------------------------------- stand-ins for unbound resources: zero buffers, a zero texel buffer, 1x1 zero images of each shape
static struct { VkBuffer zbuf; VkDeviceMemory zmem; VkBufferView ztb; VkImage img[4], dimg; VkDeviceMemory imem[4], dmem; VkImageView view[8], dview[4]; VkSampler samp; uint64_t support_off; int ok; } DZ;
static int dummies_init(void) {
    if (DZ.ok) return DZ.ok > 0;
    DZ.ok = -1;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer_(VK.dev, &bci, NULL, &DZ.zbuf)) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements_(VK.dev, DZ.zbuf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    if (ai.memoryTypeIndex == ~0u || vkAllocateMemory_(VK.dev, &ai, NULL, &DZ.zmem)) return 0;
    vkBindBufferMemory_(VK.dev, DZ.zbuf, DZ.zmem, 0);
    void *p; vkMapMemory_(VK.dev, DZ.zmem, 0, VK_WHOLE_SIZE, 0, &p); memset(p, 0, 65536); vkUnmapMemory_(VK.dev, DZ.zmem);
    VkBufferViewCreateInfo bv = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, DZ.zbuf, VK_FORMAT_R32G32B32A32_SFLOAT, 0, 4096};
    vkCreateBufferView_(VK.dev, &bv, NULL, &DZ.ztb);
    // 1D, 2D (also the 2D array view), 3D, cube-compatible 2D with 6 layers
    static const VkImageType it[4] = {VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D, VK_IMAGE_TYPE_2D};
    if (!cb_begin()) return 0;
    for (int k = 0; k < 4; k++) {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, k == 3 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, it[k], VK_FORMAT_R8G8B8A8_UNORM, {1, 1, 1}, 1, k == 3 ? 6 : 1,
                                VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage_(VK.dev, &ci, NULL, &DZ.img[k])) return 0;
        vkGetImageMemoryRequirements_(VK.dev, DZ.img[k], &mr);
        VkMemoryAllocateInfo ia = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        if (ia.memoryTypeIndex == ~0u) ia.memoryTypeIndex = vk_mem_type(mr.memoryTypeBits, 0);
        if (vkAllocateMemory_(VK.dev, &ia, NULL, &DZ.imem[k])) return 0;
        vkBindImageMemory_(VK.dev, DZ.img[k], DZ.imem[k], 0);
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, DZ.img[k], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, k == 3 ? 6 : 1}};
        vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkClearColorValue z = {{0}}; vkCmdClearColorImage_(g_cb, DZ.img[k], VK_IMAGE_LAYOUT_GENERAL, &z, 1, &b.subresourceRange);
    }
    // depth stand-ins (shadow samplers need a depth format): one D32 image of 6 layers, viewed as 2D, 2D array, cube, cube array
    {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, {1, 1, 1}, 1, 6, VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage_(VK.dev, &ci, NULL, &DZ.dimg)) return 0;
        vkGetImageMemoryRequirements_(VK.dev, DZ.dimg, &mr);
        VkMemoryAllocateInfo ia = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, vk_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        if (ia.memoryTypeIndex == ~0u) ia.memoryTypeIndex = vk_mem_type(mr.memoryTypeBits, 0);
        if (vkAllocateMemory_(VK.dev, &ia, NULL, &DZ.dmem)) return 0;
        vkBindImageMemory_(VK.dev, DZ.dimg, DZ.dmem, 0);
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, DZ.dimg, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6}};
        vkCmdPipelineBarrier_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkClearDepthStencilValue one = {1.0f, 0}; vkCmdClearDepthStencilImage_(g_cb, DZ.dimg, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &b.subresourceRange);   // depth 1: every comparison against it passes as "lit"
        static const struct { VkImageViewType t; uint32_t n; } dv[4] = {{VK_IMAGE_VIEW_TYPE_2D, 1}, {VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1}, {VK_IMAGE_VIEW_TYPE_CUBE, 6}, {VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6}};
        for (int k = 0; k < 4; k++) {
            VkImageViewCreateInfo vc = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, DZ.dimg, dv[k].t, VK_FORMAT_D32_SFLOAT, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, dv[k].n}};
            vkCreateImageView_(VK.dev, &vc, NULL, &DZ.dview[k]);
        }
    }
    full_barrier();
    // view index: SPIR-V Dim (0 1D, 1 2D, 2 3D, 3 cube) + 4 × arrayed
    static const struct { int img; VkImageViewType t; uint32_t n; } vt[8] = {{0, VK_IMAGE_VIEW_TYPE_1D, 1}, {1, VK_IMAGE_VIEW_TYPE_2D, 1}, {2, VK_IMAGE_VIEW_TYPE_3D, 1}, {3, VK_IMAGE_VIEW_TYPE_CUBE, 6},
                                                                              {0, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1}, {1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1}, {2, VK_IMAGE_VIEW_TYPE_3D, 1}, {3, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6}};
    for (int k = 0; k < 8; k++) {
        VkImageViewCreateInfo vc = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, DZ.img[vt[k].img], vt[k].t, VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, vt[k].n}};
        vkCreateImageView_(VK.dev, &vc, NULL, &DZ.view[k]);
    }
    DZ.samp = sampler_for(NULL);
    DZ.ok = 1;
    return 1;
}
static VkImageView dummy_view(uint8_t img) {
    int dim = img & 15, arr = (img >> 4) & 1; if (dim > 3) dim = 1;
    if ((img & 64) && (dim == 1 || dim == 3)) return DZ.dview[(dim == 3 ? 2 : 0) + arr];   // a depth image (shadow sampling)
    return DZ.view[dim + 4 * arr];
}

// The translator's support buffer (Ryujinx SupportBuffer.cs; nvn_mtl.m support_buf): render scale 1.0 everywhere, no BGRA swaps; written into the ring once per submission
// The support buffer: render scales 1.0, and the alpha test's function and reference (tools/spv_alpha.py), one block per (function, reference) in a submission
#define SUPPORT_SIZE 1408
static uint32_t g_at_func; static float g_at_ref; static uint64_t g_support_n;
static uint64_t support_buffer(void) {
    static uint32_t last_f; static float last_r; static uint64_t last_n = ~0ull;
    float ref = g_at_func ? g_at_ref : 0.0f;
    if (DZ.support_off != ~0ull && DZ.support_off && last_n == g_support_n && last_f == g_at_func && last_r == ref) return DZ.support_off;
    uint8_t b[SUPPORT_SIZE]; memset(b, 0, sizeof b);
    for (int i = 0; i < 73; i++) { float one = 1.0f; memcpy(b + 192 + 16 * i, &one, 4); }
    memcpy(b + 1392, &g_at_func, 4); memcpy(b + 1396, &ref, 4);
    DZ.support_off = ring_bytes(b, sizeof b); last_f = g_at_func; last_r = ref; last_n = g_support_n;
    return DZ.support_off;
}
// The support buffer with resolution scales (Ryujinx SupportBuffer render_scale, at byte 192, a float per 16 bytes): [0] the render targets' scale (fragment shaders divide
// their fragment coordinates by it), [1 + i] the scale of the stage's i-th texture (texel fetch coordinates are multiplied by it, texture sizes divided), then its images.
// frag_scale_count (byte 176) stays 0: each stage has its own support buffer here, so a vertex shader's textures start at [1] as well
static uint64_t g_support_sc;
static uint64_t support_buffer_sc(float rt, const float *sc, int n) {
    uint8_t b[SUPPORT_SIZE]; memset(b, 0, sizeof b);
    float ref = g_at_func ? g_at_ref : 0.0f;
    for (int i = 0; i < 73; i++) { float v = i == 0 ? rt : i - 1 < n ? sc[i - 1] : 1.0f; memcpy(b + 192 + 16 * i, &v, 4); }
    memcpy(b + 1392, &g_at_func, 4); memcpy(b + 1396, &ref, 4);
    g_support_sc++;
    return ring_bytes(b, sizeof b);
}

// ---------------------------------------------------------------- descriptor pools, one chain per submission
#define DP_MAX 64
static VkDescriptorPool g_dp_store[3][DP_MAX], *g_dp = g_dp_store[0]; static int g_ndp, g_dp_cur;   // the recording slot's pools (the slot in flight has the other array)
static VkDescriptorSet desc_alloc(VkDescriptorSetLayout l) {
    for (;;) {
        if (g_dp_cur >= g_ndp) {
            if (g_ndp == DP_MAX) return VK_NULL_HANDLE;
            VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32768}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8192},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 2048},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32768}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 32768}};
            VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 8192, sizeof ps / sizeof *ps, ps};
            if (vkCreateDescriptorPool_(VK.dev, &ci, NULL, &g_dp[g_ndp])) return VK_NULL_HANDLE;
            g_ndp++;
        }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_dp[g_dp_cur], 1, &l};
        VkDescriptorSet s;
        if (!vkAllocateDescriptorSets_(VK.dev, &ai, &s)) return s;
        g_dp_cur++;
    }
}

// ---------------------------------------------------------------- the bound state (NVN's command buffer state, kept across submissions as the queue keeps it)
enum { NST = 6 };   // host shader stages (hle_nvn.c st_host): 0 vertex 1 fragment 2 geometry 3 tess control 4 tess evaluation 5 compute
static const uint8_t k_nvn2model[NST] = {VKS_VS, VKS_FS, VKS_GS, VKS_TCS, VKS_TES, VKS_CS};
static const uint16_t k_c0_ssbo[NST] = {DMSH_C0_SSBO_VS, DMSH_C0_SSBO_PS, DMSH_C0_SSBO_GS, 0xffff, 0xffff, DMSH_C0_SSBO_CS};
static struct {
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t vattr[32]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[NST][18], ssbo[NST][16];
    uint64_t tex[NST][256], samp[NST][128], img[NST][32];
    float vp[16][4]; float dr[2]; p_scissor sc; int has_sc;
    uint32_t st_vmask[2], st_wmask[2], st_ref[2]; p_poly_offset po; float blend_color[4];
    int depth_clamp, prim_restart; uint32_t restart_index; int rast_discard; float alpha_ref;
    int init;
} D;
static void state_init(void) {
    if (D.init) return;
    D.init = 1;
    for (int i = 0; i < 8; i++) D.blend[i] = (blend_st){.target = (uint64_t)i, .src_rgb = NVN_BLEND_FUNC_ONE, .dst_rgb = NVN_BLEND_FUNC_ZERO, .src_a = NVN_BLEND_FUNC_ONE, .dst_a = NVN_BLEND_FUNC_ZERO,
                                                    .eq_rgb = NVN_BLEND_EQUATION_ADD, .eq_a = NVN_BLEND_EQUATION_ADD};
    D.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS};
    D.chmask.mask = 0xffffffffu;
    D.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    D.ds = (ds_st){.dt = 0, .dw = 1, .df = NVN_DEPTH_FUNC_LESS, .st = 0, .sf_f = NVN_STENCIL_FUNC_ALWAYS, .sf_b = NVN_STENCIL_FUNC_ALWAYS,
                   .fail_f = NVN_STENCIL_OP_KEEP, .zfail_f = NVN_STENCIL_OP_KEEP, .zpass_f = NVN_STENCIL_OP_KEEP, .fail_b = NVN_STENCIL_OP_KEEP, .zfail_b = NVN_STENCIL_OP_KEEP, .zpass_b = NVN_STENCIL_OP_KEEP};
    D.dr[0] = 0; D.dr[1] = 1;
    for (int f = 0; f < 2; f++) { D.st_vmask[f] = D.st_wmask[f] = 0xff; D.st_ref[f] = 0; }
    D.restart_index = 0xffffffffu;
}
static void handles_fresh(const p_bind_handles *b) { if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); }
// State records (everything mtl_exec does not handle itself)
static void state_op(uint32_t op, const uint8_t *a) {
    state_init();
    switch (op) {
    case OP_BIND_STATE: {
        const p_bind_state *b = (const p_bind_state *)a;
        switch (b->kind) {
        case 0: { blend_st s; memcpy(&s, b->bytes, sizeof s); D.blend[s.target & 7] = s; break; }
        case 1: memcpy(&D.color, b->bytes, sizeof D.color); break;
        case 2: memcpy(&D.chmask, b->bytes, sizeof D.chmask); break;
        case 3: memcpy(&D.ms, b->bytes, sizeof D.ms); break;
        case 4: memcpy(&D.poly, b->bytes, sizeof D.poly); break;
        case 5: memcpy(&D.ds, b->bytes, sizeof D.ds); break;
        }
        break;
    }
    case OP_BIND_VATTR: { const p_bind_vattr *b = (const p_bind_vattr *)a; if (b->n >= 0 && b->n <= 32) { memcpy(D.vattr, b->st, 4 * b->n); D.nattr = b->n; } break; }
    case OP_BIND_VSTREAM: { const p_bind_vstream *b = (const p_bind_vstream *)a; if (b->n >= 0 && b->n <= 16) { memcpy(D.stream, b->st, 8 * b->n); D.nstream = b->n; } break; }
    case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tmp; const p_bind_vbuf *b = vbuf_rec(op, a, &tmp); if (b->first >= 0 && b->first + b->n <= 16) memcpy(&D.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
    case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tmp; const p_bind_buf *b = bind_buf_rec(op, a, &tmp); if ((unsigned)b->stage < NST && (unsigned)b->index < 18) D.ubo[b->stage][b->index] = *b; break; }
    case OP_BIND_SSBO: { const p_bind_buf *b = (const p_bind_buf *)a; if ((unsigned)b->stage < NST && (unsigned)b->index < 16) D.ssbo[b->stage][b->index] = *b; break; }
    case OP_BIND_TEX: case OP_BIND_TEX1: { p_bind_handles t1; const p_bind_handles *b = bind_rec(op, a, &t1);
        if (b->n < 0 || b->n > 32) break;
        handles_fresh(b);
        if ((unsigned)b->stage < NST && b->first >= 0 && b->first + b->n <= 256) memcpy(&D.tex[b->stage][b->first], b->h, 8 * b->n);
        break; }
    case OP_BIND_SAMP: { const p_bind_handles *b = (const p_bind_handles *)a; if (b->n < 0 || b->n > 32) break; handles_fresh(b);
        if ((unsigned)b->stage < NST && b->first >= 0 && b->first + b->n <= 128) memcpy(&D.samp[b->stage][b->first], b->h, 8 * b->n); break; }
    case OP_BIND_IMG: { const p_bind_handles *b = (const p_bind_handles *)a; if (b->n < 0 || b->n > 32) break; handles_fresh(b);
        if ((unsigned)b->stage < NST && b->first >= 0 && b->first + b->n <= 32) memcpy(&D.img[b->stage][b->first], b->h, 8 * b->n); break; }
    case OP_STENCIL: { const p_stencil *b = (const p_stencil *)a;
        for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) D.st_vmask[f] = b->value_mask; else if (b->which == 1) D.st_wmask[f] = b->mask; else D.st_ref[f] = b->ref; }
        break; }
    case OP_POLY_OFFSET: D.po = *(const p_poly_offset *)a; break;
    case OP_VIEWPORTS: { const p_viewports *b = (const p_viewports *)a; for (int i = 0; i < b->n && b->first + i < 16; i++) memcpy(D.vp[b->first + i], b->v[i], 16); break; }
    case OP_SCISSOR: D.sc = *(const p_scissor *)a; D.has_sc = 1; break;
    case OP_DEPTH_RANGE: { const p_depth_range *b = (const p_depth_range *)a; D.dr[0] = b->n; D.dr[1] = b->f; break; }
    case OP_DEPTH_CLAMP: D.depth_clamp = ((const p_depth_clamp *)a)->on; break;
    case OP_BLEND_COLOR: memcpy(D.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
    case OP_ALPHA_REF: D.alpha_ref = ((const p_alpha_ref *)a)->ref; break;
    case OP_PRIM_RESTART: { const p_prim_restart *b = (const p_prim_restart *)a; D.prim_restart = b->on != 0; D.restart_index = b->index; break; }
    case OP_RAST_DISCARD: D.rast_discard = *(const int32_t *)a != 0; break;
    default: break;
    }
}

// ---------------------------------------------------------------- statistics (in the replay's summary line)
enum { SK_NOPROG, SK_NORT, SK_TESS, SK_PIPE, SK_RING, SK_PRIM, SK_DESC, SK_N };
static const char *k_sk[SK_N] = {"shaders not ready", "no render target", "tessellation / geometry off", "pipeline refused", "upload ring full", "primitive type", "descriptors"};
static uint64_t g_drawn, g_disp_done, g_skip[SK_N], g_pipes, g_unsup_sbuf_ubo, g_tex_missing, g_tex_unres, g_tex_unreg, g_storage_unsup;

// ---------------------------------------------------------------- the render pass (dynamic rendering) over the current render targets
static struct { int open; VkFormat cf[8]; VkFormat df; uint32_t ncol, w, h, samples; int has_d, has_s; vtex *cv[8], *dv; uint32_t clv[8], cl0[8], dlv, dl0; uint64_t draws0, cser[8]; VkFormat cvf[8]; vtex *clr[10]; int nclr; uint32_t cdirty; int ddirty; float sc; } P;   // sc: the targets' resolution scale   // clr: attachments this pass cleared (load op or in-pass clear)
static void dump_pass(void);
static int g_dump_subs, g_dump_pass, g_dump_lines, g_dump_light;
static vtex *g_dump_tex[64]; static int g_ndump_tex; static uint64_t g_dump_saved[512]; static int g_ndump_saved;   // sampled textures to save after the pass (once per dump)   // frame dumps (below)
static void prof_mark(int begin, uint32_t w, uint32_t h, uint32_t ncol, uint32_t depth, uint32_t compute, uint64_t fs, uint64_t draws);
static uint64_t g_pass_fs;
// Render target clears go into the render pass (a tile-based GPU such as the Adreno then starts its tiles from the clear value instead of loading the whole target from
// memory, and a clear between draws no longer splits the pass). Outside a pass a clear waits here until the next pass begins: one of its attachments takes it as its load
// operation. Anything else that ends a pass (copies, dispatches, the submission's end) does the waiting clears first, as image clears
typedef struct { vtex *v; uint32_t lv, l0, nlay; VkImageAspectFlags asp; float rgba[4]; float depth; uint32_t stencil; int done; } pclear;
static pclear g_pcl[24]; static int g_npcl; static uint64_t g_clear_load, g_clear_att, g_clear_img, g_clear_why[128], g_clear_untaken, g_clear_split; static uint32_t g_cur_op;   // why: the op that made waiting clears image clears
static void pcl_image(const pclear *c) {
    VkImageSubresourceRange rr = {c->asp, c->lv, 1, c->l0, c->nlay};
    if (c->asp & VK_IMAGE_ASPECT_COLOR_BIT) {
        VkClearColorValue cv; memcpy(cv.float32, c->rgba, 16);
        if (fmt_int(c->v->fmt)) for (int i = 0; i < 4; i++) cv.uint32[i] = (uint32_t)c->rgba[i];
        vkCmdClearColorImage_(g_cb, c->v->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rr);
    } else { VkClearDepthStencilValue z = {c->depth, c->stencil}; vkCmdClearDepthStencilImage_(g_cb, c->v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    g_clear_img++;
}
static void clears_flush(void) {
    if (!g_npcl) return;
    g_clear_why[g_cur_op & 127] += g_npcl;
    int any = 0;
    for (int i = 0; i < g_npcl; i++) if (!g_pcl[i].done && g_pcl[i].v->img) { pcl_image(&g_pcl[i]); any = 1; }
    g_npcl = 0;
    if (any) full_barrier();
}
static void pass_end(void) { if (P.open) { vkCmdEndRendering_(g_cb); prof_mark(0, P.w, P.h, P.ncol, P.dv != NULL, 0, g_pass_fs, g_drawn - P.draws0); full_barrier(); P.open = 0; dump_pass(); } else clears_flush(); }
static VkClearValue pcl_value(const pclear *c, VkFormat vf) {
    VkClearValue cv; memset(&cv, 0, sizeof cv);
    if (c->asp & VK_IMAGE_ASPECT_COLOR_BIT) { memcpy(cv.color.float32, c->rgba, 16); if (fmt_int(vf)) for (int i = 0; i < 4; i++) cv.color.uint32[i] = (uint32_t)c->rgba[i]; }
    else { cv.depthStencil.depth = c->depth; cv.depthStencil.stencil = c->stencil; }
    return cv;
}
static int lvl_is(const vtex *v, uint32_t lv, uint32_t w, uint32_t h) { return (v->w >> lv ? v->w >> lv : 1) == w && (v->h >> lv ? v->h >> lv : 1) == h; }
// A clear of target v (level lv, layers l0..l0+nlay): into the open pass, or kept for the next one. 0: not taken (the caller ends the pass and clears the image)
static int clear_defer(vtex *v, uint32_t lv, uint32_t l0, uint32_t nlay, VkImageAspectFlags asp, const float *rgba, float depth, uint32_t stencil) {
    // off unless DM_VK_CLEAR_IN_PASS=1: with it on, Mario Tennis Fever's picture went black after a while on the Thor (Turnip), cause not found yet (no pass read a
    // cleared target as a texture)
    static int off = -1; if (off < 0) { const char *e = getenv("DM_VK_CLEAR_IN_PASS"); off = !(e && *e == '1'); }
    if (off || v->is3d || nlay != 1) return 0;
    pclear c = {v, lv, l0, nlay, asp, {0}, depth, stencil, 0}; if (rgba) memcpy(c.rgba, rgba, 16);
    if (P.open) {
        VkClearAttachment ca = {asp, 0, pcl_value(&c, VK_FORMAT_UNDEFINED)};
        int ok = 0;
        if (asp & VK_IMAGE_ASPECT_COLOR_BIT) {
            for (uint32_t i = 0; i < P.ncol && i < 8; i++) if (P.cv[i] == v && P.clv[i] == lv && P.cl0[i] == l0) { ca.colorAttachment = i; ca.clearValue = pcl_value(&c, P.cf[i]); ok = 1; break; }
        } else ok = P.dv == v && P.dlv == lv && P.dl0 == l0 && (!(asp & VK_IMAGE_ASPECT_DEPTH_BIT) || P.has_d) && (!(asp & VK_IMAGE_ASPECT_STENCIL_BIT) || P.has_s);
        if (ok && lvl_is(v, lv, P.w, P.h)) {
            VkClearRect r = {{{0, 0}, {P.w, P.h}}, 0, 1};
            vkCmdClearAttachments_(g_cb, 1, &ca, 1, &r); g_clear_att++; g_cb_ops++; if (P.nclr < 10) P.clr[P.nclr++] = v;
            return 1;
        }
        pass_end();
    }
    for (int i = 0; i < g_npcl; i++)   // the same clear again: the later value
        if (g_pcl[i].v == v && g_pcl[i].lv == lv && g_pcl[i].l0 == l0 && g_pcl[i].asp == asp) { g_pcl[i] = c; return 1; }
    if (g_npcl == (int)(sizeof g_pcl / sizeof *g_pcl)) clears_flush();
    g_pcl[g_npcl++] = c;
    return 1;
}
// At a pass's start (attachments known, before vkCmdBeginRendering): the waiting clears its attachments take as their load operation; the rest are done as image clears.
// All of one image's clears go the same way, so they stay in order
static void clears_into_pass(VkRenderingAttachmentInfo *ca, VkRenderingAttachmentInfo *da, VkRenderingAttachmentInfo *sa, uint32_t w, uint32_t h) {
    if (!g_npcl) return;
    int take[24];
    for (int i = 0; i < g_npcl; i++) {
        pclear *c = &g_pcl[i]; take[i] = 0;
        if (!lvl_is(c->v, c->lv, w, h)) continue;
        if (c->asp & VK_IMAGE_ASPECT_COLOR_BIT) { for (uint32_t k = 0; k < P.ncol && k < 8; k++) if (P.cv[k] == c->v && P.clv[k] == c->lv && P.cl0[k] == c->l0) take[i] = 1 + k; }
        else if (P.dv == c->v && P.dlv == c->lv && P.dl0 == c->l0 && (!(c->asp & VK_IMAGE_ASPECT_DEPTH_BIT) || P.has_d) && (!(c->asp & VK_IMAGE_ASPECT_STENCIL_BIT) || P.has_s)) take[i] = 100;
    }
    for (int i = 0; i < g_npcl; i++) if (!take[i]) for (int j = 0; j < g_npcl; j++) if (g_pcl[j].v == g_pcl[i].v) take[j] = 0;
    int img = 0;
    for (int i = 0; i < g_npcl; i++) {
        pclear *c = &g_pcl[i];
        if (!take[i]) { if (c->v->img) { pcl_image(c); img = 1; g_clear_untaken++; } continue; }
        if (take[i] == 100) {
            if (c->asp & VK_IMAGE_ASPECT_DEPTH_BIT) { da->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da->clearValue.depthStencil.depth = c->depth; }
            if (c->asp & VK_IMAGE_ASPECT_STENCIL_BIT) { sa->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa->clearValue.depthStencil.stencil = c->stencil; }
        } else { ca[take[i] - 1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca[take[i] - 1].clearValue = pcl_value(c, P.cf[take[i] - 1]); }
        g_clear_load++; if (P.nclr < 10) P.clr[P.nclr++] = c->v;
    }
    g_npcl = 0;
    if (img) full_barrier();
}
static uint64_t g_fb_split;
// DM_VK_FEEDBACK_SPLIT: 0 off, 1 (default) when a draw of the pass wrote the target, 2 when the pass drew anything at all
static int fb_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_FEEDBACK_SPLIT"); on = e && *e >= '0' && *e <= '2' ? *e - '0' : 1; } return on; }
static void clear_color(const p_clear_color *c) {
    if (!R.has_rt || c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) return;
    if (P.open) { vtex *pv = tex_get(R.rt.color[c->index]); if (!pv || !(pv->bad & 2)) pass_end(); }   // its first use records commands outside a pass
    g_rt_hint = 1; vtex *v = use(R.rt.color[c->index]); g_rt_hint = 0; if (!v) return;
    gpu_wrote(v);
    static int warned; if (c->mask != 0xf && !warned++) LOG("[vk] a colour clear with channel mask %#x was done on all channels (stage B has no masked clears)", c->mask);
    uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.cview[c->index], R.rt.has_cview[c->index], &lv, &nl, &l0, &nlay);
    if (clear_defer(v, lv, l0, nlay, VK_IMAGE_ASPECT_COLOR_BIT, c->rgba, 0, 0)) return;
    pass_end();
    pclear pc = {v, lv, l0, nlay, VK_IMAGE_ASPECT_COLOR_BIT, {0}, 0, 0, 0}; memcpy(pc.rgba, c->rgba, 16);
    pcl_image(&pc); full_barrier();
}
static void clear_ds(const p_clear_ds *c) {
    if (!R.has_rt || !R.rt.depth) return;
    if (P.open) { vtex *pv = tex_get(R.rt.depth); if (!pv || !(pv->bad & 2)) pass_end(); }
    g_rt_hint = 1; vtex *v = use(R.rt.depth); g_rt_hint = 0; if (!v) return;
    gpu_wrote(v);
    VkImageAspectFlags a = fmt_aspects(v->fmt) & ((c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask ? VK_IMAGE_ASPECT_STENCIL_BIT : 0));
    if (!a) return;
    uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.dview, R.rt.has_dview, &lv, &nl, &l0, &nlay);
    if (clear_defer(v, lv, l0, nlay, a, NULL, c->depth, (uint32_t)c->stencil)) return;
    pass_end();
    pclear pc = {v, lv, l0, nlay, a, {0}, c->depth, (uint32_t)c->stencil, 0};
    pcl_image(&pc); full_barrier();
}
// A texture a draw reads, ready for it: its first use (layout, zero fill) and its guest content are command-buffer work outside a render pass, so a pass open then is ended first
static vtex *prep_tex(const tex_t *t) {
    vtex *v = tex_get(t); if (!v) return NULL;
    if (v->uploaded && cpu_wrote(v)) { v->uploaded = 0; g_reup++; }   // the CPU wrote its storage since the upload
    if (!(v->bad & 2) || ((!v->gpu || v->linear) && !v->uploaded && t->mem && v->samples == 1)) { pass_end(); if (!cb_begin()) return NULL; img_general(v); tex_fill(v, t); }
    return v;
}

// ---------------------------------------------------------------- GPU profile: timestamps around each render pass and dispatch, summed per kind of pass (size, targets,
// first fragment shader) and logged with the replay summary: where the GPU's time goes. DM_VK_PROFILE=0 turns it off
#define PQ_CAP 2048
// DM_VK_ABTEST=gs: draws with a geometry stage are skipped every other profile period (the profile says which), to measure what they cost
static int g_ab_skip_gs, g_ab = -1;
static VkQueryPool g_pq; static uint32_t g_npq; static int g_prof = -1;
static uint32_t g_gs_vps = 4;   // viewports a passthrough geometry shader's draw goes to without it (DM_SPV_VIEWPORTS, as spv_passthrough.py)
typedef struct { uint64_t key, fs; uint32_t w, h, ncol, depth, compute; double ms; uint64_t n, draws; } pkind;
static pkind g_pk[512]; static int g_npk;
typedef struct { uint32_t q0; uint64_t key, fs; uint32_t w, h, ncol, depth, compute; uint64_t draws; } pmark;
static pmark g_pm_store[3][PQ_CAP / 2], *g_pm = g_pm_store[0]; static int g_npm; static pthread_mutex_t g_prof_mu = PTHREAD_MUTEX_INITIALIZER; static double g_prof_total_ms, g_prof_gap_ms; static uint64_t g_prof_subs;
static uint32_t g_pm_cur = ~0u;
static int prof_on(void) {
    if (g_prof < 0) { const char *e = getenv("DM_VK_PROFILE"); g_prof = !(e && *e == '0') && VK.limits.timestampPeriod > 0; }
    if (g_prof && !g_pq) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, PQ_CAP, 0};
        if (vkCreateQueryPool_(VK.dev, &qi, NULL, &g_pq)) { g_prof = 0; return 0; }
    }
    return g_prof;
}
static void prof_cb_begin(void) {   // at a command buffer's start: the queries reset, query 0 the start
    if (!prof_on()) return;
    vkCmdResetQueryPool_(g_cb, g_pq, 0, PQ_CAP); vkCmdWriteTimestamp_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pq, 0); g_npq = 1; g_npm = 0;
}
static void prof_mark(int begin, uint32_t w, uint32_t h, uint32_t ncol, uint32_t depth, uint32_t compute, uint64_t fs, uint64_t draws) {
    if (!g_prof || !g_pq || g_npq + 2 >= PQ_CAP) return;
    if (begin) {
        g_pm_cur = g_npm < PQ_CAP / 2 ? (uint32_t)g_npm++ : ~0u; if (g_pm_cur == ~0u) return;
        g_pm[g_pm_cur] = (pmark){g_npq, 0, fs, w, h, ncol, depth, compute, 0};
        vkCmdWriteTimestamp_(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pq, g_npq++);
    } else if (g_pm_cur != ~0u) {
        if (fs && !g_pm[g_pm_cur].fs) g_pm[g_pm_cur].fs = fs;
        g_pm[g_pm_cur].draws = draws;
        vkCmdWriteTimestamp_(g_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pq, g_npq++); g_pm_cur = ~0u;
    }
}
static void prof_cb_end(void) { if (g_prof && g_pq && g_npq < PQ_CAP) vkCmdWriteTimestamp_(g_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pq, g_npq++); }
static void prof_collect(VkQueryPool pq, uint32_t npq, const pmark *pm, int npm) {   // after a submission completed (its slot's queries; on the completion thread)
    if (!g_prof || !pq || npq < 2) return;
    static uint64_t ts[PQ_CAP];
    if (vkGetQueryPoolResults_(VK.dev, pq, 0, npq, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT)) return;
    pthread_mutex_lock(&g_prof_mu);
    double per = VK.limits.timestampPeriod / 1e6;
    g_prof_total_ms += (ts[npq - 1] - ts[0]) * per; g_prof_subs++;
    // the GPU's time from the end of the submission before to the start of this one: other work on the GPU (the window's present, the compositor) or idle
    static uint64_t last_end; if (last_end && ts[0] > last_end) { double gap = (ts[0] - last_end) * per; if (gap < 200) g_prof_gap_ms += gap; }
    last_end = ts[npq - 1];
    for (int i = 0; i < npm; i++) {
        const pmark *m = &pm[i]; if (m->q0 + 1 >= npq) continue;
        double ms = (ts[m->q0 + 1] - ts[m->q0]) * per; if (ms < 0 || ms > 1000) continue;
        uint64_t key = (uint64_t)m->w << 40 ^ (uint64_t)m->h << 24 ^ m->ncol << 4 ^ m->depth << 2 ^ m->compute ^ m->fs * 31;
        int k = 0; while (k < g_npk && g_pk[k].key != key) k++;
        if (k == g_npk) { if (g_npk >= 512) continue; g_pk[g_npk++] = (pkind){key, m->fs, m->w, m->h, m->ncol, m->depth, m->compute, 0, 0, 0}; }
        g_pk[k].ms += ms; g_pk[k].n++; g_pk[k].draws += m->draws;
    }
    pthread_mutex_unlock(&g_prof_mu);
}
static void prof_report(void) {
    if (!g_prof) return;
    pthread_mutex_lock(&g_prof_mu);
    if (!g_prof_subs) { pthread_mutex_unlock(&g_prof_mu); return; }
    double sum = 0; for (int i = 0; i < g_npk; i++) sum += g_pk[i].ms;
    LOG("[vk] GPU profile (last %llu submissions): %.2f ms GPU per submission, %.2f ms of it in passes and dispatches; between submissions (other GPU work or idle) %.2f ms; the most expensive kinds:", (unsigned long long)g_prof_subs, g_prof_total_ms / g_prof_subs, sum / g_prof_subs, g_prof_gap_ms / g_prof_subs);
    for (int r = 0; r < 14; r++) {
        int b = -1; for (int i = 0; i < g_npk; i++) if (g_pk[i].n && (b < 0 || g_pk[i].ms > g_pk[b].ms)) b = i;
        if (b < 0) break;
        pkind *k = &g_pk[b];
        LOG("[vk]   %5.2f ms per submission: %s %ux%u, %u colour%s, %llu times, %.0f draws each, first fragment shader %016llx", k->ms / g_prof_subs, k->compute ? "dispatch" : "pass", k->w, k->h, k->ncol, k->depth ? " + depth" : "",
            (unsigned long long)k->n, k->n ? (double)k->draws / k->n : 0.0, (unsigned long long)k->fs);
        k->n = 0;
    }
    g_npk = 0; g_prof_total_ms = 0; g_prof_gap_ms = 0; g_prof_subs = 0;
    pthread_mutex_unlock(&g_prof_mu);
    if (g_ab < 0) { const char *e = getenv("DM_VK_ABTEST"); g_ab = e && !strcmp(e, "gs"); }
    if (g_ab) { g_ab_skip_gs = !g_ab_skip_gs; LOG("[vk] A/B test: the next period %s draws with a geometry stage", g_ab_skip_gs ? "SKIPS" : "draws"); }
}

static int pass_begin(void) {
    if (P.open) return 1;
    if (!R.has_rt) return 0;
    VkRenderingAttachmentInfo ca[8], da, sa; uint32_t nc = 0, w = ~0u, h = ~0u, samples = 0; int any = 0;
    memset(&P, 0, sizeof P); P.sc = 1.0f;
    for (int i = 0; i < R.rt.n && i < 8; i++) {
        ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
        P.cf[i] = VK_FORMAT_UNDEFINED; nc = i + 1;
        const tex_t *t = R.rt.color[i]; if (!t) continue;
        g_rt_hint = 1; vtex *v = prep_tex(t); g_rt_hint = 0; if (!v || fmt_depth(v->fmt)) continue;
        uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.cview[i], R.rt.has_cview[i], &lv, &nl, &l0, &nlay);
        VkFormat vf = tex_view_fmt2(v, t, &R.rt.cview[i], R.rt.has_cview[i]);
        if (R.rt.has_cview[i] && R.rt.cview[i].set_format) {   // which targets a game draws through a view of another format (logged once per texture and format, up to 40)
            static uint64_t seen[40]; static int ns; uint64_t k = t->serial << 16 | R.rt.cview[i].format; int j = 0; while (j < ns && seen[j] != k) j++;
            if (j == ns && ns < 40) { seen[ns++] = k; LOG("[vk] render target serial %llu (NVN format %#x, %ux%u) drawn through a view of NVN format %#x: Vulkan format %d (image format %d)",
                (unsigned long long)t->serial, t->d.format, t->d.width, t->d.height, R.rt.cview[i].format, vf, v->fmt); }
        }
        VkImageView iv = view_get_fmt(v, vf, v->is3d ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, lv, 1, v->is3d ? 0 : l0, 1, (VkComponentMapping){0, 0, 0, 0}, VK_IMAGE_ASPECT_COLOR_BIT, 2);
        if (!iv) continue;
        if (v->sc != 1.0f) P.sc = v->sc;
        gpu_wrote(v); ca[i].imageView = iv; P.cf[i] = vf; P.cser[i] = t->serial; P.cvf[i] = vf; any = 1; P.cv[i] = v; P.clv[i] = lv; P.cl0[i] = v->is3d ? 0 : l0;
        uint32_t lw = v->w >> lv ? v->w >> lv : 1, lh = v->h >> lv ? v->h >> lv : 1;
        if (lw < w) w = lw;
        if (lh < h) h = lh;
        samples = v->samples;
    }
    P.ncol = nc;
    da = sa = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                          VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
    P.df = VK_FORMAT_UNDEFINED;
    if (R.rt.depth) {
        g_rt_hint = 1; vtex *v = prep_tex(R.rt.depth); g_rt_hint = 0;
        if (v && fmt_depth(v->fmt)) {
            uint32_t lv, nl, l0, nlay; view_range(v, &R.rt.dview, R.rt.has_dview, &lv, &nl, &l0, &nlay);
            VkImageView iv = view_get(v, VK_IMAGE_VIEW_TYPE_2D, lv, 1, l0, 1, (VkComponentMapping){0, 0, 0, 0}, fmt_aspects(v->fmt), 2);
            if (iv) {
                if (v->sc != 1.0f) P.sc = v->sc;
                gpu_wrote(v); P.df = v->fmt; any = 1; P.dv = v; P.dlv = lv; P.dl0 = l0;
                VkImageAspectFlags a = fmt_aspects(v->fmt);
                if (a & VK_IMAGE_ASPECT_DEPTH_BIT) { da.imageView = iv; P.has_d = 1; }
                if (a & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.imageView = iv; P.has_s = 1; }
                uint32_t lw = v->w >> lv ? v->w >> lv : 1, lh = v->h >> lv ? v->h >> lv : 1;
                if (lw < w) w = lw;
                if (lh < h) h = lh;
                if (!samples) samples = v->samples;
            }
        }
    }
    {   // attachments of different resolution scales (one scaled, one not): logged, the first few (the pass is drawn at the scaled size)
        int ns = 0, nu = 0; for (uint32_t i = 0; i < nc && i < 8; i++) if (P.cv[i]) { if (P.cv[i]->sc != 1.0f) ns++; else nu++; }
        if (P.dv) { if (P.dv->sc != 1.0f) ns++; else nu++; }
        static int nm; if (ns && nu && nm++ < 10) LOG("[YELLOW] [vk] a pass mixes scaled and unscaled targets (colour serials %llu %llu %llu %llu, depth %s): resolution scaling breaks it",
            (unsigned long long)P.cser[0], (unsigned long long)P.cser[1], (unsigned long long)P.cser[2], (unsigned long long)P.cser[3], P.dv ? (P.dv->sc != 1.0f ? "scaled" : "not scaled") : "none");
    }
    if (!any) { clears_flush(); return 0; }
    if (!cb_begin()) return 0;
    clears_into_pass(ca, &da, &sa, w, h);
    P.w = w; P.h = h; P.samples = samples ? samples : 1; P.draws0 = g_drawn;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, 1, 0, nc, ca, P.has_d ? &da : NULL, P.has_s ? &sa : NULL};
    g_pass_fs = 0; prof_mark(1, w, h, nc, P.dv != NULL, 0, 0, 0);
    vkCmdBeginRendering_(g_cb, &ri);
    P.open = 1; g_cb_ops++;
    return 1;
}

// ---------------------------------------------------------------- pipelines
static VkFormat vfmt(uint32_t f) {
    switch (f) {
    case NVN_FORMAT_R8: return VK_FORMAT_R8_UNORM; case NVN_FORMAT_R8SN: return VK_FORMAT_R8_SNORM; case NVN_FORMAT_R8UI: return VK_FORMAT_R8_UINT; case NVN_FORMAT_R8I: return VK_FORMAT_R8_SINT;
    case NVN_FORMAT_R16F: return VK_FORMAT_R16_SFLOAT; case NVN_FORMAT_R16: return VK_FORMAT_R16_UNORM; case NVN_FORMAT_R16SN: return VK_FORMAT_R16_SNORM; case NVN_FORMAT_R16UI: return VK_FORMAT_R16_UINT; case NVN_FORMAT_R16I: return VK_FORMAT_R16_SINT;
    case NVN_FORMAT_R32F: return VK_FORMAT_R32_SFLOAT; case NVN_FORMAT_R32UI: return VK_FORMAT_R32_UINT; case NVN_FORMAT_R32I: return VK_FORMAT_R32_SINT;
    case NVN_FORMAT_RG8: return VK_FORMAT_R8G8_UNORM; case NVN_FORMAT_RG8SN: return VK_FORMAT_R8G8_SNORM; case NVN_FORMAT_RG8UI: return VK_FORMAT_R8G8_UINT; case NVN_FORMAT_RG8I: return VK_FORMAT_R8G8_SINT;
    case NVN_FORMAT_RG16F: return VK_FORMAT_R16G16_SFLOAT; case NVN_FORMAT_RG16: return VK_FORMAT_R16G16_UNORM; case NVN_FORMAT_RG16SN: return VK_FORMAT_R16G16_SNORM; case NVN_FORMAT_RG16UI: return VK_FORMAT_R16G16_UINT; case NVN_FORMAT_RG16I: return VK_FORMAT_R16G16_SINT;
    case NVN_FORMAT_RG32F: return VK_FORMAT_R32G32_SFLOAT; case NVN_FORMAT_RG32UI: return VK_FORMAT_R32G32_UINT; case NVN_FORMAT_RG32I: return VK_FORMAT_R32G32_SINT;
    case NVN_FORMAT_RGB8: return VK_FORMAT_R8G8B8_UNORM; case NVN_FORMAT_RGB8SN: return VK_FORMAT_R8G8B8_SNORM; case NVN_FORMAT_RGB8UI: return VK_FORMAT_R8G8B8_UINT; case NVN_FORMAT_RGB8I: return VK_FORMAT_R8G8B8_SINT;
    case NVN_FORMAT_RGB16F: return VK_FORMAT_R16G16B16_SFLOAT; case NVN_FORMAT_RGB16: return VK_FORMAT_R16G16B16_UNORM; case NVN_FORMAT_RGB16SN: return VK_FORMAT_R16G16B16_SNORM; case NVN_FORMAT_RGB16UI: return VK_FORMAT_R16G16B16_UINT; case NVN_FORMAT_RGB16I: return VK_FORMAT_R16G16B16_SINT;
    case NVN_FORMAT_RGB32F: return VK_FORMAT_R32G32B32_SFLOAT; case NVN_FORMAT_RGB32UI: return VK_FORMAT_R32G32B32_UINT; case NVN_FORMAT_RGB32I: return VK_FORMAT_R32G32B32_SINT;
    case NVN_FORMAT_RGBA8: case NVN_FORMAT_RGBX8: return VK_FORMAT_R8G8B8A8_UNORM; case NVN_FORMAT_RGBA8SN: case NVN_FORMAT_RGBX8SN: return VK_FORMAT_R8G8B8A8_SNORM;
    case NVN_FORMAT_RGBA8UI: case NVN_FORMAT_RGBX8UI: return VK_FORMAT_R8G8B8A8_UINT; case NVN_FORMAT_RGBA8I: case NVN_FORMAT_RGBX8I: return VK_FORMAT_R8G8B8A8_SINT;
    case NVN_FORMAT_RGBA16F: case NVN_FORMAT_RGBX16F: return VK_FORMAT_R16G16B16A16_SFLOAT; case NVN_FORMAT_RGBA16: case NVN_FORMAT_RGBX16: return VK_FORMAT_R16G16B16A16_UNORM;
    case NVN_FORMAT_RGBA16SN: case NVN_FORMAT_RGBX16SN: return VK_FORMAT_R16G16B16A16_SNORM; case NVN_FORMAT_RGBA16UI: case NVN_FORMAT_RGBX16UI: return VK_FORMAT_R16G16B16A16_UINT;
    case NVN_FORMAT_RGBA16I: case NVN_FORMAT_RGBX16I: return VK_FORMAT_R16G16B16A16_SINT;
    case NVN_FORMAT_RGBA32F: case NVN_FORMAT_RGBX32F: return VK_FORMAT_R32G32B32A32_SFLOAT; case NVN_FORMAT_RGBA32UI: case NVN_FORMAT_RGBX32UI: return VK_FORMAT_R32G32B32A32_UINT;
    case NVN_FORMAT_RGBA32I: case NVN_FORMAT_RGBX32I: return VK_FORMAT_R32G32B32A32_SINT;
    case NVN_FORMAT_RGB10A2: return VK_FORMAT_A2B10G10R10_UNORM_PACK32; case NVN_FORMAT_RGB10A2UI: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case NVN_FORMAT_RGB10A2SN: return VK_FORMAT_A2B10G10R10_SNORM_PACK32; case NVN_FORMAT_RGB10A2I: return VK_FORMAT_A2B10G10R10_SINT_PACK32;
    case NVN_FORMAT_RGB10A2_UI2F: return VK_FORMAT_A2B10G10R10_USCALED_PACK32; case NVN_FORMAT_RGB10A2_I2F: return VK_FORMAT_A2B10G10R10_SSCALED_PACK32;
    case NVN_FORMAT_R11G11B10F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32; case NVN_FORMAT_BGRA8: case NVN_FORMAT_BGRX8: return VK_FORMAT_B8G8R8A8_UNORM;
    case NVN_FORMAT_R8_UI2F: return VK_FORMAT_R8_USCALED; case NVN_FORMAT_R8_I2F: return VK_FORMAT_R8_SSCALED; case NVN_FORMAT_R16_UI2F: return VK_FORMAT_R16_USCALED; case NVN_FORMAT_R16_I2F: return VK_FORMAT_R16_SSCALED;
    case NVN_FORMAT_RG8_UI2F: return VK_FORMAT_R8G8_USCALED; case NVN_FORMAT_RG8_I2F: return VK_FORMAT_R8G8_SSCALED; case NVN_FORMAT_RG16_UI2F: return VK_FORMAT_R16G16_USCALED; case NVN_FORMAT_RG16_I2F: return VK_FORMAT_R16G16_SSCALED;
    case NVN_FORMAT_RGB8_UI2F: return VK_FORMAT_R8G8B8_USCALED; case NVN_FORMAT_RGB8_I2F: return VK_FORMAT_R8G8B8_SSCALED; case NVN_FORMAT_RGB16_UI2F: return VK_FORMAT_R16G16B16_USCALED; case NVN_FORMAT_RGB16_I2F: return VK_FORMAT_R16G16B16_SSCALED;
    case NVN_FORMAT_RGBA8_UI2F: return VK_FORMAT_R8G8B8A8_USCALED; case NVN_FORMAT_RGBA8_I2F: return VK_FORMAT_R8G8B8A8_SSCALED; case NVN_FORMAT_RGBA16_UI2F: return VK_FORMAT_R16G16B16A16_USCALED; case NVN_FORMAT_RGBA16_I2F: return VK_FORMAT_R16G16B16A16_SSCALED;
    case NVN_FORMAT_R32_UI2F: case NVN_FORMAT_R32_I2F: case NVN_FORMAT_RG32_UI2F: case NVN_FORMAT_RG32_I2F: case NVN_FORMAT_RGB32_UI2F: case NVN_FORMAT_RGB32_I2F: case NVN_FORMAT_RGBA32_UI2F: case NVN_FORMAT_RGBA32_I2F:
        return VK_FORMAT_UNDEFINED;   // 32-bit integers read as floats: Vulkan has no scaled 32-bit formats
    default: return VK_FORMAT_UNDEFINED;
    }
}
// An integer vertex format read by a float shader input: its SCALED twin gives the same numbers as floats (the Switch converts them so; Vulkan wants the
// format's numeric kind to match the input). 32-bit integers have no twin and stay as they are
static VkFormat vfmt_for_input(VkFormat f, int ct) {
    if (ct != 0) return f;
    int base = -1, n = 7;
    if (f >= VK_FORMAT_R8_UNORM && f <= VK_FORMAT_A8B8G8R8_SRGB_PACK32) base = 9;
    else if (f >= VK_FORMAT_A2R10G10B10_UNORM_PACK32 && f <= VK_FORMAT_A2B10G10R10_SINT_PACK32) base = 58, n = 6;
    else if (f >= VK_FORMAT_R16_UNORM && f <= VK_FORMAT_R16G16B16A16_SFLOAT) base = 70;
    if (base < 0) return f;
    int k = (f - base) % n;
    return k == 4 || k == 5 ? (VkFormat)(f - 2) : f;   // UINT → USCALED, SINT → SSCALED
}
static int vfmt_ok(VkFormat f) { if (!f) return 0; VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties_(VK.phys, f, &fp); return (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0; }
static VkPrimitiveTopology topo_of(int prim) {
    switch (prim) {
    case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case NVN_DRAW_PRIMITIVE_LINE_LOOP: case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_PATCHES: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;   // a convex polygon is a fan
    case NVN_DRAW_PRIMITIVE_QUADS: case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // no Vulkan topology: indices made per draw (quad_indices)
    default: return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
    }
}
static const VkBlendFactor k_bf[0x65] = {
    [NVN_BLEND_FUNC_ZERO] = VK_BLEND_FACTOR_ZERO, [NVN_BLEND_FUNC_ONE] = VK_BLEND_FACTOR_ONE, [NVN_BLEND_FUNC_SRC_COLOR] = VK_BLEND_FACTOR_SRC_COLOR,
    [NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR] = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, [NVN_BLEND_FUNC_SRC_ALPHA] = VK_BLEND_FACTOR_SRC_ALPHA, [NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    [NVN_BLEND_FUNC_DST_ALPHA] = VK_BLEND_FACTOR_DST_ALPHA, [NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, [NVN_BLEND_FUNC_DST_COLOR] = VK_BLEND_FACTOR_DST_COLOR,
    [NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR] = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, [NVN_BLEND_FUNC_SRC_ALPHA_SATURATE] = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
    [NVN_BLEND_FUNC_SRC1_COLOR] = VK_BLEND_FACTOR_SRC1_COLOR, [NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR, [NVN_BLEND_FUNC_SRC1_ALPHA] = VK_BLEND_FACTOR_SRC1_ALPHA,
    [NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, [NVN_BLEND_FUNC_CONSTANT_COLOR] = VK_BLEND_FACTOR_CONSTANT_COLOR,
    [NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, [NVN_BLEND_FUNC_CONSTANT_ALPHA] = VK_BLEND_FACTOR_CONSTANT_ALPHA,
    [NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
static VkBlendFactor bf(unsigned f) { return f < 0x65 && (f == NVN_BLEND_FUNC_ZERO || k_bf[f]) ? k_bf[f] : VK_BLEND_FACTOR_ONE; }
static VkBlendOp beq(unsigned e) { return e >= 1 && e <= 5 ? (VkBlendOp)(e - 1) : VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
static VkStencilOp stop(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }

// What a graphics pipeline depends on, all of it compared byte for byte
typedef struct {
    uint64_t prog;
    VkFormat cf[8], df; uint32_t ncol, samples;
    uint32_t topo, front_ccw, depth_gl, rast_discard, depth_clamp, prim_restart;
    blend_st blend[8]; uint32_t blend_en, chmask; ds_st ds; poly_st poly; uint8_t a2c, atest, nogs, pad;
    struct { uint8_t loc, binding, used, pad; uint32_t fmt, offset; } va[32]; uint32_t nva;
    struct { uint32_t stride, rate; } vb[17]; uint32_t vb_used;
} pkey;
typedef struct { pkey k; VkPipeline p; uint64_t hash; int used; } pentry;   // a refused pipeline is kept too (p null), so it is not tried again at every draw
#define PC_CAP (1u << 14)
static pentry *g_pc; static uint32_t g_npc;
static VkPipelineCache g_pcache;
static uint64_t pkey_hash(const pkey *k) { uint64_t h = 1469598103934665603ull; const uint8_t *p = (const uint8_t *)k; for (size_t i = 0; i < sizeof *k; i++) h = (h ^ p[i]) * 1099511628211ull; return h; }
static VkPipeline pipe_make(const vks_prog *pr, const pkey *k) {
    VkPipelineShaderStageCreateInfo st[5]; uint32_t ns = 0;
    // the alpha test's switch (tools/spv_alpha.py: a bool specialization constant, SpecId 200; a shader without it ignores the entry)
    static const VkSpecializationMapEntry at_me = {200, 0, 4}; VkBool32 at_on = k->atest ? VK_TRUE : VK_FALSE;
    VkSpecializationInfo at_si = {1, &at_me, 4, &at_on};
    for (int m = 0; m < VKS_NGFX; m++) if (pr->mod[m] && !(m == VKS_GS && k->nogs))
        st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, (VkShaderStageFlagBits)(1u << (m == VKS_FS ? 4 : m)), pr->mod[m], "main", m == VKS_FS && k->atest ? &at_si : NULL};
    VkVertexInputBindingDescription vb[17]; uint32_t nvb = 0;
    for (uint32_t b = 0; b < 17; b++) if (k->vb_used >> b & 1) vb[nvb++] = (VkVertexInputBindingDescription){b, k->vb[b].stride, (VkVertexInputRate)k->vb[b].rate};
    VkVertexInputAttributeDescription va[32]; uint32_t nva = 0;
    for (uint32_t i = 0; i < k->nva; i++) va[nva++] = (VkVertexInputAttributeDescription){k->va[i].loc, k->va[i].binding, (VkFormat)k->va[i].fmt, k->va[i].offset};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, nvb, vb, nva, va};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo,
        k->prim_restart && k->topo != VK_PRIMITIVE_TOPOLOGY_POINT_LIST && k->topo != VK_PRIMITIVE_TOPOLOGY_LINE_LIST && k->topo != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, VK_TRUE};
    // a geometry stage may pick the viewport (Mario Tennis Fever's shadow cascades: 4): all 16 then; otherwise 1
    uint32_t nvp = pr->st[VKS_GS] && !k->nogs && VK.limits.maxViewports > 1 ? (VK.limits.maxViewports < 16 ? VK.limits.maxViewports : 16) : 1;
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->depth_gl ? &dcc : NULL, 0, nvp, NULL, nvp, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, k->rast_discard,
        k->poly.mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : k->poly.mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL,
        (VkCullModeFlags)(k->poly.cull & 3), k->front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE, k->poly.offset_en != 0, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, 0, 0, NULL, k->a2c, 0};
    const ds_st *d = &k->ds;
    VkStencilOpState sf = {stop(d->fail_f), stop(d->zpass_f), stop(d->zfail_f), cmpop(d->sf_f), 0xff, 0xff, 0}, sb = {stop(d->fail_b), stop(d->zpass_b), stop(d->zfail_b), cmpop(d->sf_b), 0xff, 0xff, 0};
    int hd = k->df && (fmt_aspects(k->df) & VK_IMAGE_ASPECT_DEPTH_BIT), hs = k->df && (fmt_aspects(k->df) & VK_IMAGE_ASPECT_STENCIL_BIT);
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, hd && d->dt, hd && d->dt && d->dw, cmpop(d->df), 0, hs && d->st, sf, sb, 0, 1};
    VkPipelineColorBlendAttachmentState cb[8];
    for (uint32_t i = 0; i < k->ncol; i++) {
        const blend_st *b = &k->blend[i];
        int en = (k->blend_en >> i & 1) && k->cf[i] && !fmt_int(k->cf[i]);
        cb[i] = (VkPipelineColorBlendAttachmentState){en, bf(b->src_rgb), bf(b->dst_rgb), beq(b->eq_rgb), bf(b->src_a), bf(b->dst_a), beq(b->eq_a), (k->chmask >> (4 * i)) & 0xf};
    }
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, VK_LOGIC_OP_COPY, k->ncol, cb, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                            VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, k->cf, hd ? k->df : VK_FORMAT_UNDEFINED, hs ? k->df : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, ns, st, &vi, &ia, NULL, &vp, &rs, &ms, &dss, &cbs, &ds, pr->layout, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    if (!g_pcache) { VkPipelineCacheCreateInfo pc = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, 0, NULL}; vkCreatePipelineCache_(VK.dev, &pc, NULL, &g_pcache); }
    VkPipeline p = VK_NULL_HANDLE;
    if (pr->st[VKS_GS]) { static int n; if (n++ < 30) { LOG("[vk] making a pipeline with geometry shader %016llx (vertex %016llx, fragment %016llx)",
        (unsigned long long)pr->st[VKS_GS]->key, pr->st[VKS_VS] ? (unsigned long long)pr->st[VKS_VS]->key : 0ull, pr->st[VKS_FS] ? (unsigned long long)pr->st[VKS_FS]->key : 0ull); fflush(stderr); } }
    VkResult r = vkCreateGraphicsPipelines_(VK.dev, g_pcache, 1, &ci, NULL, &p);
    if (r) {
        static int n; if (n++ < 10) {
            char va_s[256] = ""; int o = 0;
            for (uint32_t i = 0; i < k->nva && o < 220; i++) o += snprintf(va_s + o, sizeof va_s - o, " %u:%u@%u/b%u", k->va[i].loc, k->va[i].fmt, k->va[i].offset, k->va[i].binding);
            LOG("[RED] [vk] a graphics pipeline was refused (%d): shaders %016llx / %016llx%s; colour formats %d %d %d %d (%u), depth format %d, %u samples, topology %u, cull %u mode %u, "
                "depth test %u write %u func %u bias %u clamp %u, stencil %u, blend %#x mask %#x, a2c %u, discard %u, GL depth %u; inputs%s", r,
                pr->st[VKS_VS] ? (unsigned long long)pr->st[VKS_VS]->key : 0ull, pr->st[VKS_FS] ? (unsigned long long)pr->st[VKS_FS]->key : 0ull, pr->st[VKS_GS] ? " +geometry" : "",
                k->cf[0], k->cf[1], k->cf[2], k->cf[3], k->ncol, k->df, k->samples, k->topo, (unsigned)k->poly.cull, (unsigned)k->poly.mode, (unsigned)k->ds.dt, (unsigned)k->ds.dw, (unsigned)k->ds.df,
                (unsigned)k->poly.offset_en, k->depth_clamp, (unsigned)k->ds.st, k->blend_en, k->chmask, k->a2c, k->rast_discard, k->depth_gl, va_s);
        }
        return VK_NULL_HANDLE;
    }
    g_pipes++;
    return p;
}
static VkPipeline pipe_get(const vks_prog *pr, const pkey *k) {
    static pkey last; static VkPipeline last_p;   // the same state as the draw before (most draws): no hash
    if (last_p && !memcmp(&last, k, sizeof *k)) return last_p;
    if (!g_pc) g_pc = calloc(PC_CAP, sizeof *g_pc);
    VkPipeline p0 = VK_NULL_HANDLE;
    #define PG_RET(x) do { p0 = (x); if (p0) { last = *k; last_p = p0; } return p0; } while (0)
    uint64_t h = pkey_hash(k); uint32_t i = (uint32_t)(h >> 40) & (PC_CAP - 1);
    for (uint32_t n = 0; n < PC_CAP; n++, i = (i + 1) & (PC_CAP - 1)) {
        if (!g_pc[i].used) break;
        if (g_pc[i].hash == h && !memcmp(&g_pc[i].k, k, sizeof *k)) PG_RET(g_pc[i].p);
    }
    VkPipeline p = pipe_make(pr, k);
    if (g_npc < PC_CAP * 3 / 4 && !g_pc[i].used) { g_pc[i].k = *k; g_pc[i].p = p; g_pc[i].hash = h; g_pc[i].used = 1; g_npc++; }
    PG_RET(p);
    #undef PG_RET
}
// What the command buffer has bound already (per command buffer: cb_begin clears it): the same pipeline, vertex buffers, index buffer, viewports, scissors and descriptor
// sets as the draw before are not bound again (each bind is driver work on the CPU, about 1,650 draws a frame)
static struct { VkPipeline pipe; VkDeviceSize vo[17]; VkBuffer vb[17]; uint32_t vset; VkDeviceSize io; VkIndexType it; int ib; uint32_t nvp; VkViewport vp[16]; VkRect2D sc[16];
                VkPipelineLayout lay; VkDescriptorSet sets[5]; uint32_t nset, ndo, dofs[128]; uint64_t saved; } BS;
static void bs_reset(void) { memset(&BS, 0, sizeof BS); }

// ---------------------------------------------------------------- descriptors: one set for a shader, from the bound state of its NVN stage
// A texture binding's handle words (spv_pack.py): a word in the driver table c2 names a slot (combined BindTexture slots, separate textures, images, separate samplers);
// a word in a uniform buffer (bank 3 + n) holds the handle itself (bindless)
static uint32_t c2_word(int st, uint32_t byte, int *ok) {
    *ok = 1;
    if (byte >= DMSH_C2_TEX && byte < DMSH_C2_TEX + DMSH_C2_ROW * 32 && !((byte - DMSH_C2_TEX) % 8)) return (uint32_t)D.tex[st][(byte - DMSH_C2_TEX) / 8];
    if (byte >= DMSH_C2_IMG && byte < DMSH_C2_IMG + DMSH_C2_ROW * DMSH_C2_IMG_SLOTS && !((byte - DMSH_C2_IMG) % 8)) return (uint32_t)D.img[st][(byte - DMSH_C2_IMG) / 8];
    if (byte >= DMSH_C2_SEPTEX && byte < DMSH_C2_SAMP && !((byte - DMSH_C2_SEPTEX) % 8)) return (uint32_t)D.tex[st][DMSH_SEPTEX_SLOT0 + (byte - DMSH_C2_SEPTEX) / 8];
    if (byte >= DMSH_C2_SAMP && byte < DMSH_C2_SAMP + DMSH_C2_ROW * DMSH_C2_SEP_SLOTS && !((byte - DMSH_C2_SAMP) % 8)) return (uint32_t)D.samp[st][(byte - DMSH_C2_SAMP) / 8];
    *ok = 0; return 0;
}
static uint32_t handle_of(int st, const vks_bind *b, int *ok) {
    uint32_t h = b->reg, type = h >> 28, w0 = (h & 0x3fff) * 4, w1 = ((h >> 14) & 0x3fff) * 4;
    *ok = 0;
    if (b->aux == 2 || b->aux == 0xffff) {   // the translator gives the driver table as bank -1 (its default texture bank, NVN's c2)
        int o0, o1 = 1; uint32_t v = c2_word(st, w0, &o0);
        if (type == 1) v |= c2_word(st, w1, &o1);
        *ok = o0 && o1; return v;
    }
    if (b->aux >= 3 && b->aux - 3 < 18) {   // bindless: the handle words are in the game's uniform buffer
        const p_bind_buf *u = &D.ubo[st][b->aux - 3];
        if (!u->addr || w0 + 4 > u->size || (type == 1 && w1 + 4 > u->size)) return 0;
        pend_hazard(u->addr + w0, 4); if (type == 1) pend_hazard(u->addr + w1, 4);
        uint32_t v; memcpy(&v, (const void *)(uintptr_t)(u->addr + w0), 4);
        if (type == 1) { uint32_t s2; memcpy(&s2, (const void *)(uintptr_t)(u->addr + w1), 4); v |= s2; }
        *ok = 1; return v;
    }
    return 0;
}
typedef struct { VkWriteDescriptorSet w[256]; VkDescriptorBufferInfo bi[512]; VkDescriptorImageInfo ii[512]; VkBufferView tv[512]; uint32_t nw, nbi, nii, ntv; int dyn; uint32_t ndo; struct { uint32_t binding, off; } dofs[64]; } wset;   // dofs: the dynamic uniform buffers' offsets
static wset g_ws;

// Texture buffers (NVN_TEXTURE_TARGET_BUFFER): the texture's storage (width texels of its format) copied into the ring like a uniform buffer, and a buffer view of
// the format over it, made for this submission (destroyed when its slot is reused: slot_clean). Written ones (shader images of buffer textures) are copied back with the submission
static VkBufferView *g_tbv; static int g_ntbv, g_ctbv; static uint64_t g_tbuf_n, g_tbuf_miss;
static VkBufferView tbuf_view(int st, const vks_bind *b, int storage) {
    int ok = 0; uint32_t h = handle_of(st, b, &ok);
    if (!ok) { g_tex_unres++; return VK_NULL_HANDLE; }
    const treg *r = storage ? ireg_get((int)(h & 0xfffff)) : treg_get((int)(h & 0xfffff));
    if (!r || r->t.d.target != NVN_TEXTURE_TARGET_BUFFER || !r->t.mem) { g_tbuf_miss++; return VK_NULL_HANDLE; }
    VkFormat f = tex_vkfmt(&r->t); int bw, bh, bpb = nvn_fmt_bpb(r->t.d.format, &bw, &bh);
    if (!f || bpb <= 0) { g_tbuf_miss++; return VK_NULL_HANDLE; }
    static uint8_t okc[256], okv[256];   // the format's texel buffer support, asked once
    if (f < 256 && !okc[f]) { okc[f] = 1; VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties_(VK.phys, f, &fp);
        okv[f] = (fp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT ? 1 : 0) | (fp.bufferFeatures & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT ? 2 : 0);
        if (!(okv[f] & 1)) LOG("[RED] [vk] format %d (NVN %#x) cannot be a texture buffer on this GPU: those read zero", f, r->t.d.format); }
    if (f >= 256 || !(okv[f] & (storage ? 2 : 1))) { g_tbuf_miss++; return VK_NULL_HANDLE; }
    uint64_t elems = r->t.d.width ? r->t.d.width : 1;
    if (elems > VK.limits.maxTexelBufferElements) elems = VK.limits.maxTexelBufferElements;
    uint64_t size = elems * (uint64_t)bpb;
    uint64_t off = ring_put((uint64_t)(uintptr_t)r->t.mem, size); if (off == ~0ull) return (VkBufferView)(uintptr_t)~0ull;
    if (storage) wb_add((uint64_t)(uintptr_t)r->t.mem, size, off);
    VkBufferViewCreateInfo bv = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, RG.b, f, off, size};
    VkBufferView view = VK_NULL_HANDLE;
    if (vkCreateBufferView_(VK.dev, &bv, NULL, &view)) { g_tbuf_miss++; return VK_NULL_HANDLE; }
    if (g_ntbv == g_ctbv) { g_ctbv = g_ctbv ? 2 * g_ctbv : 256; g_tbv = realloc(g_tbv, g_ctbv * sizeof *g_tbv); }
    g_tbv[g_ntbv++] = view; g_tbuf_n++;
    static int once; if (!once++) LOG("[vk] texture buffers: on (first: NVN format %#x, %llu texels)", r->t.d.format, (unsigned long long)elems);
    return view;
}
static int write_set(const vks_shader *s, int st, VkDescriptorSet set) {
    wset *W = &g_ws; W->nw = W->nbi = W->nii = W->ntv = 0; W->ndo = 0;
    int sup_bi = -1; float bsc[64]; uint8_t bkind[64]; memset(bkind, 0, sizeof bkind);   // the support buffer's descriptor; each texture / image binding's resolution scale
    for (int i = 0; i < s->nb; i++) { const vks_bind *b = &s->b[i];   // every texture and image the translator indexes (not arrays, not separate samplers), scale 1 until known
        if (b->binding < 64 && (b->count <= 1) && b->vktype != VK_DESCRIPTOR_TYPE_SAMPLER && (b->kind == VKS_TEX || b->kind == VKS_IMG)) { bkind[b->binding] = (uint8_t)b->kind; bsc[b->binding] = 1.0f; } }
    for (int i = 0; i < s->nb && W->nw < 256; i++) {
        const vks_bind *b = &s->b[i]; uint32_t n = b->count ? b->count : 1;
        VkWriteDescriptorSet *w = &W->w[W->nw++];
        *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, b->binding, 0, n, (VkDescriptorType)b->vktype, NULL, NULL, NULL};
        switch (b->vktype) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
            if (W->nbi + n > 512) return 0;
            VkDescriptorBufferInfo *bi = &W->bi[W->nbi]; W->nbi += n; w->pBufferInfo = bi;
            int uni = b->vktype == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; uint64_t lim = uni ? VK.limits.maxUniformBufferRange : (64ull << 20);
            if (lim > 65536 && uni) lim = 65536;
            uint64_t addr = 0, size = 0, off = ~0ull;
            if (b->kind == VKS_SUPPORT) off = support_buffer(), size = SUPPORT_SIZE, sup_bi = W->nbi - (int)n;
            else if (b->kind == VKS_CBUF) {
                if (b->reg == 1 && s->c1_len) { off = ring_put((uint64_t)(uintptr_t)s->c1, s->c1_len); size = (s->c1_len + 15) & ~15u; }
                else if (b->reg >= 3 && b->reg - 3 < 18) { addr = D.ubo[st][b->reg - 3].addr; size = D.ubo[st][b->reg - 3].size; }
            } else if (b->kind == VKS_SBUF) {
                if (b->aux == 0 && k_c0_ssbo[st] != 0xffff && b->reg >= k_c0_ssbo[st] && (b->reg - k_c0_ssbo[st]) % DMSH_C0_ROW == 0 && (b->reg - k_c0_ssbo[st]) / DMSH_C0_ROW < 16) {
                    const p_bind_buf *q = &D.ssbo[st][(b->reg - k_c0_ssbo[st]) / DMSH_C0_ROW]; addr = q->addr; size = q->size;
                } else g_unsup_sbuf_ubo++;
            }
            if (addr && size) {
                if (size > lim) size = lim;
                off = ring_put(addr, size);   // the guest's exact bytes; the range rounds up into the ring's slack
                if (off != ~0ull && b->kind == VKS_SBUF && (b->image & 1)) wb_add(addr, size, off);
                size = (size + 15) & ~15ull;
            }
            if (off == ~0ull && addr && size) return -1;   // the ring is full
            for (uint32_t k = 0; k < n; k++) bi[k] = off != ~0ull ? (VkDescriptorBufferInfo){RG.b, off, size} : (VkDescriptorBufferInfo){DZ.zbuf, 0, 65536};
            if (W->dyn && uni) {   // dynamic: the offset goes in at bind time (one per array element, in binding order)
                w->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                for (uint32_t k = 0; k < n && W->ndo < 64; k++) { W->dofs[W->ndo].binding = b->binding << 8 | k; W->dofs[W->ndo++].off = (uint32_t)bi[k].offset; bi[k].offset = 0; }
                if (b->kind == VKS_SUPPORT) sup_bi = -2 - (int)(W->ndo - n);   // its dynamic offset's index (patched below for resolution scales)
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {   // texture buffers: the guest's bytes in the ring, seen through a buffer view
            if (W->ntv + n > 512) return 0;
            VkBufferView *tv = &W->tv[W->ntv]; W->ntv += n; w->pTexelBufferView = tv;
            VkBufferView view = tbuf_view(st, b, b->vktype == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);
            if (view == (VkBufferView)(uintptr_t)~0ull) return -1;   // the ring is full
            for (uint32_t k = 0; k < n; k++) tv[k] = view ? view : DZ.ztb;
            break;
        }
        default: {   // samplers, sampled images, combined image samplers, storage images
            if (W->nii + n > 512) return 0;
            VkDescriptorImageInfo *ii = &W->ii[W->nii]; W->nii += n; w->pImageInfo = ii;
            int storage = b->vktype == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ok = 0;
            uint32_t h = handle_of(st, b, &ok);
            VkImageView iv = VK_NULL_HANDLE; VkSampler sp = DZ.samp;
            if (ok) {
                const treg *r = storage ? ireg_get((int)(h & 0xfffff)) : treg_get((int)(h & 0xfffff));
                if (b->vktype != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE && b->vktype != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) { VkSampler x = sampler_for(sreg_get((int)(h >> 20))); if (x) sp = x; }
                int why = 0;   // 1 no image (format or size refused), 2 the format cannot be a shader image (storage), 3 no view of the shader's image type
                if (r && b->vktype != VK_DESCRIPTOR_TYPE_SAMPLER) {
                    vtex *v = prep_tex(&r->t);
                    // a target this pass cleared, also read by its shaders: a tile-based GPU has the clear only in its tiles, the texture reads memory. The pass ends (its
                    // clear stored) and begins again loading it, so the shaders read the cleared target as with an image clear (Mario Tennis Fever's picture went black)
                    if (v && P.open) for (int c = 0; c < P.nclr; c++) if (P.clr[c] == v) { pass_end(); g_clear_split++; break; }
                    // a target this pass has drawn into, read by this draw's shaders (a particle reading the depth, a target read back): what the GPU wrote may still
                    // sit in its caches (or compressed, or in a tile), and the texture reads memory. The pass ends (written out) and begins again. On the Thor's GPU,
                    // passes rendered straight to memory (TU_DEBUG=sysmem, and render target clears in passes, which change the driver's choice) read garbage there
                    // and Mario Tennis Fever's picture went black. NVN orders such reads with a texture barrier, which the replay records as nothing
                    // Only when a draw of this pass wrote it (dirty: the colour channels or the depth / stencil writes on): particles reading the depth they test against,
                    // with depth writes off, do not end the pass each
                    if (v && P.open && fb_on()) {
                        int any = fb_on() == 2 && g_drawn > P.draws0;
                        int hit = P.dv == v && (P.ddirty || any); for (uint32_t c = 0; c < P.ncol && c < 8 && !hit; c++) hit = P.cv[c] == v && ((P.cdirty >> c & 1) || any);
                        if (hit) { pass_end(); g_fb_split++; }
                    }
                    if (b->binding < 64 && bkind[b->binding] && v) bsc[b->binding] = v->sc;
                    if (v && storage && v->sc != 1.0f) { static int ns; if (ns++ < 10) LOG("[YELLOW] [vk] shader %016llx binds the scaled render target serial %llu as a shader image (binding %u): its writes land at unscaled texels",
                                                                   (unsigned long long)s->key, (unsigned long long)r->t.serial, b->binding); }
                    if (!v) why = 1;
                    else if (storage && !v->storage) why = 2;
                    else { iv = tex_view(v, &r->t, &r->v, r->has_v, b->image, storage); if (storage && iv) gpu_wrote(v); if (!iv) why = 3; }
                }
                if (!iv && r) {
                    g_tex_missing++; if (why == 2) g_storage_unsup++;
                    // each texture logged once (up to 40): a scene's missing pieces show up even when one texture misses at every draw
                    static uint64_t seen[40]; static int nseen; int k = 0; while (k < nseen && seen[k] != r->t.serial) k++;
                    if (k == nseen && nseen < 40) { seen[nseen++] = r->t.serial;
                        LOG("[vk] shader %016llx binding %u: texture serial %llu (NVN format %#x target %d %ux%u, %u samples, flags %#x) %s (shader image type %#x%s)",
                            (unsigned long long)s->key, b->binding, (unsigned long long)r->t.serial, r->t.d.format, r->t.d.target, r->t.d.width, r->t.d.height, r->t.d.samples, r->t.d.flags,
                            why == 1 ? "has no Vulkan image" : why == 2 ? "cannot be a shader image in this format (written by the shader: its output is lost)" : "has no view of the shader's image type",
                            b->image, storage ? ", storage" : ""); }
                }
                if (!r && b->vktype != VK_DESCRIPTOR_TYPE_SAMPLER && (h & 0xfffff)) { g_tex_unreg++; static int n; if (n++ < 10) LOG("[vk] shader %016llx binding %u: %s ID %u is not registered", (unsigned long long)s->key, b->binding, storage ? "image" : "texture", h & 0xfffff); }
            } else { g_tex_unres++; static int n; if (n++ < 10) LOG("[vk] shader %016llx binding %u: handle %#x in bank %u is not one this layer reads (texture slots, images, separate samplers; uniform buffers)", (unsigned long long)s->key, b->binding, b->reg, b->aux); }
            if (g_dump_subs && g_dump_lines < 4000 && b->vktype != VK_DESCRIPTOR_TYPE_SAMPLER) {
                g_dump_lines++;
                const treg *r = ok ? (storage ? ireg_get((int)(h & 0xfffff)) : treg_get((int)(h & 0xfffff))) : NULL;
                if (iv && r) { vtex *v2 = tex_get(&r->t);
                    if (v2 && !v2->gpu && v2->w <= 1024 && v2->h <= 1024 && g_ndump_tex < 64) {   // a texture from the game's files: saved once per dump, after the pass
                        int seen = 0; for (int q = 0; q < g_ndump_saved && !seen; q++) seen = g_dump_saved[q] == v2->serial;
                        for (int q = 0; q < g_ndump_tex && !seen; q++) seen = g_dump_tex[q] == v2;
                        if (!seen) g_dump_tex[g_ndump_tex++] = v2;
                    }
                    LOG("[vk] dump:   pass %d shader %016llx binding %u%s: texture serial %llu (%ux%u, NVN format %#x, read as format %d from the image of serial %llu format %d)", g_dump_pass, (unsigned long long)s->key, b->binding, storage ? " (image)" : "",
                        (unsigned long long)r->t.serial, r->t.d.width, r->t.d.height, r->t.d.format, v2 ? tex_view_fmt(v2, &r->t) : 0, v2 ? (unsigned long long)v2->serial : 0ull, v2 ? v2->fmt : 0); }
                else LOG("[vk] dump:   pass %d shader %016llx binding %u%s: EMPTY (%s; handle %#x bank %u)", g_dump_pass, (unsigned long long)s->key, b->binding, storage ? " (image)" : "",
                         !ok ? "handle not decoded" : !r ? "ID not registered" : "no usable view", b->reg, b->aux);
            }
            if (!iv) iv = dummy_view(b->image);
            for (uint32_t k = 0; k < n; k++) ii[k] = (VkDescriptorImageInfo){sp, iv, VK_IMAGE_LAYOUT_GENERAL};
            break;
        }
        }
    }
    // resolution scaling: this stage's own support buffer when a scale is not 1 (the translator's order: textures by binding, then images)
    float rt = s->model == VKS_FS && P.sc > 0 ? P.sc : 1.0f; int scaled = rt != 1.0f;
    for (int i = 0; i < 64 && !scaled; i++) scaled = bkind[i] && bsc[i] != 1.0f;
    if (scaled && sup_bi >= 0) {
        float sc[72]; int n = 0;
        for (int i = 0; i < 64 && n < 72; i++) if (bkind[i] == VKS_TEX) sc[n++] = bsc[i];
        for (int i = 0; i < 64 && n < 72; i++) if (bkind[i] == VKS_IMG) sc[n++] = bsc[i];
        uint64_t off = support_buffer_sc(rt, sc, n);
        if (off != ~0ull) W->bi[sup_bi].offset = off;
    } else if (scaled && sup_bi <= -2) {
        float sc[72]; int n = 0;
        for (int i = 0; i < 64 && n < 72; i++) if (bkind[i] == VKS_TEX) sc[n++] = bsc[i];
        for (int i = 0; i < 64 && n < 72; i++) if (bkind[i] == VKS_IMG) sc[n++] = bsc[i];
        uint64_t off = support_buffer_sc(rt, sc, n);
        if (off != ~0ull) W->dofs[-2 - sup_bi].off = (uint32_t)off;
    }
    (void)set;
    return 1;
}
// Descriptor sets reused within a submission: a set's content (everything but the dynamic uniform buffer offsets) hashed with its layout; the same content again
// binds the set already written. Draws mostly differ only in their uniform data, so with dynamic uniform buffers most need no new set (each new one cost a pool
// allocation and a descriptor update, a large part of the replay's time per draw)
#define DSC_CAP 8192
static struct { uint64_t h; VkDescriptorSet set; } g_dsc[DSC_CAP]; static uint32_t g_dsc_n; static uint64_t g_dsc_hit, g_dsc_miss;
static _Atomic int g_dsc_stale;   // a texture was released (its views are gone): no set made before is reused
static void dsc_reset(void) { if (g_dsc_n) { memset(g_dsc, 0, sizeof g_dsc); g_dsc_n = 0; } }
static uint64_t ws_hash(VkDescriptorSetLayout l) {
    wset *W = &g_ws; uint64_t h = 1469598103934665603ull ^ (uint64_t)(uintptr_t)l;
    #define HM(x) do { uint64_t x_ = (uint64_t)(x); h ^= x_; h *= 1099511628211ull; h ^= h >> 29; } while (0)
    for (uint32_t i = 0; i < W->nw; i++) {
        const VkWriteDescriptorSet *w = &W->w[i]; HM(w->dstBinding); HM(w->descriptorType); HM(w->descriptorCount);
        for (uint32_t k = 0; k < w->descriptorCount; k++) {
            if (w->pBufferInfo) { HM((uintptr_t)w->pBufferInfo[k].buffer); HM(w->pBufferInfo[k].offset); HM(w->pBufferInfo[k].range); }
            if (w->pImageInfo) { HM((uintptr_t)w->pImageInfo[k].imageView); HM((uintptr_t)w->pImageInfo[k].sampler); HM(w->pImageInfo[k].imageLayout); }
            if (w->pTexelBufferView) HM((uintptr_t)w->pTexelBufferView[k]);
        }
    }
    #undef HM
    return h ? h : 1;
}
static VkDescriptorSet ws_commit(VkDescriptorSetLayout l) {   // the set for what write_set gathered (reused, or allocated and written)
    if (atomic_exchange(&g_dsc_stale, 0)) dsc_reset();
    wset *W = &g_ws; uint64_t h = ws_hash(l); uint32_t i = (uint32_t)(h >> 40) & (DSC_CAP - 1);
    for (int k = 0; k < 16; k++, i = (i + 1) & (DSC_CAP - 1)) {
        if (g_dsc[i].h == h) { g_dsc_hit++; return g_dsc[i].set; }
        if (!g_dsc[i].h) break;
    }
    VkDescriptorSet set = desc_alloc(l); if (!set) return VK_NULL_HANDLE;
    for (uint32_t k = 0; k < W->nw; k++) W->w[k].dstSet = set;
    vkUpdateDescriptorSets_(VK.dev, W->nw, W->w, 0, NULL);
    g_dsc_miss++;
    if (!g_dsc[i].h && g_dsc_n < DSC_CAP * 3 / 4) { g_dsc[i].h = h; g_dsc[i].set = set; g_dsc_n++; }
    return set;
}
// The sets of a program, written and bound (graphics: one per stage, empty for absent stages; compute: set 0)
static int bind_sets(const vks_prog *pr, VkPipelineBindPoint bp) {
    VkDescriptorSet sets[5]; int nset = pr->nset; uint32_t dofs[128], ndo = 0;
    for (int k = 0; k < nset; k++) {
        const vks_shader *s = NULL;
        for (int m = 0; m < 6; m++) if (pr->st[m] && vks_set_of(m) == k && (m == VKS_CS) == (pr->compute != 0)) s = pr->st[m];
        if (!s) { g_ws.nw = 0; sets[k] = ws_commit(pr->set[k]); if (!sets[k]) return 0; continue; }
        int st = 0; for (int x = 0; x < NST; x++) if (k_nvn2model[x] == s->model) st = x;
        g_ws.dyn = pr->dyn;
        int r = write_set(s, st, VK_NULL_HANDLE); if (r <= 0) return r;
        sets[k] = ws_commit(pr->set[k]); if (!sets[k]) return 0;
        // this set's dynamic offsets, in binding order
        wset *W = &g_ws;
        for (uint32_t a = 1; a < W->ndo; a++) for (uint32_t b2 = a; b2 > 0 && W->dofs[b2 - 1].binding > W->dofs[b2].binding; b2--) { __typeof__(W->dofs[0]) t = W->dofs[b2]; W->dofs[b2] = W->dofs[b2 - 1]; W->dofs[b2 - 1] = t; }
        for (uint32_t a = 0; a < W->ndo && ndo < 128; a++) dofs[ndo++] = W->dofs[a].off;
    }
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS && BS.lay == pr->layout && BS.nset == (uint32_t)nset && BS.ndo == ndo && !memcmp(BS.sets, sets, nset * sizeof *sets) && !memcmp(BS.dofs, dofs, ndo * 4)) { BS.saved++; return 1; }
    vkCmdBindDescriptorSets_(g_cb, bp, pr->layout, 0, (uint32_t)nset, sets, ndo, ndo ? dofs : NULL);
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) { BS.lay = pr->layout; BS.nset = (uint32_t)nset; BS.ndo = ndo; memcpy(BS.sets, sets, nset * sizeof *sets); memcpy(BS.dofs, dofs, ndo * 4); }
    return 1;
}

typedef struct { int prim; int indexed; uint32_t count, instances, first, base_instance; int32_t base_vertex; uint64_t ibuf; int itype; } draw_args;
// Quads and quad strips as triangle lists: two triangles per quad, in the quad's winding (v0 v1 v2, v0 v2 v3; a strip's quad k is 2k, 2k+1, 2k+3, 2k+2).
// The indices go into the ring as 32-bit; an indexed draw's own indices are read through them (restart indices end the strip there)
static uint64_t g_quads;
static int quad_indices(int prim, const draw_args *a, uint64_t *off, uint32_t *count) {
    uint32_t nq = prim == NVN_DRAW_PRIMITIVE_QUADS ? a->count / 4 : (a->count >= 4 ? (a->count - 2) / 2 : 0);
    if (!nq) return 0;
    uint64_t o = ring_alloc((uint64_t)nq * 6 * 4); if (o == ~0ull) return -1;
    uint32_t *dst = (uint32_t *)(RG.p + o);
    uint32_t isz = !a->indexed ? 0 : a->itype == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : a->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 1;
    const uint8_t *ib = a->indexed ? (const uint8_t *)(uintptr_t)a->ibuf : NULL;
    if (ib) pend_hazard(a->ibuf, (uint64_t)a->count * isz);
    #define QI(n) (ib ? (isz == 4 ? ((const uint32_t *)ib)[n] : isz == 2 ? ((const uint16_t *)ib)[n] : ib[n]) : a->first + (n))
    for (uint32_t q = 0; q < nq; q++) {
        uint32_t v0, v1, v2, v3;
        if (prim == NVN_DRAW_PRIMITIVE_QUADS) { v0 = 4 * q; v1 = v0 + 1; v2 = v0 + 2; v3 = v0 + 3; }
        else { v0 = 2 * q; v1 = v0 + 1; v2 = v0 + 3; v3 = v0 + 2; }
        uint32_t *t = dst + 6 * q;
        t[0] = QI(v0); t[1] = QI(v1); t[2] = QI(v2); t[3] = QI(v0); t[4] = QI(v2); t[5] = QI(v3);
    }
    #undef QI
    *off = o; *count = nq * 6; g_quads++;
    return 1;
}

// ---------------------------------------------------------------- a draw
// A passthrough geometry shader's viewport mask from its program (tools/spv_gsmask.py) and the geometry stage's bound uniform buffers; 0 when it cannot be worked out
static int gs_mask(const vks_shader *s, uint32_t *mask) {
    uint32_t st[32]; int sp = 0;
    for (uint32_t i = 0; i < s->ngsm; i++) {
        uint32_t o = s->gsm[i];
        if (o == 1) { if (sp >= 32 || i + 1 >= s->ngsm) return 0; st[sp++] = s->gsm[++i]; }
        else if (o == 2) {
            if (sp < 1 || i + 1 >= s->ngsm) return 0;
            uint32_t slot = s->gsm[++i], dw = st[sp - 1];
            if (slot < 3 || slot - 3 >= 18) return 0;
            int gst = 0; for (int x = 0; x < NST; x++) if (k_nvn2model[x] == s->model) gst = x;   // the NVN stage of the geometry shader
            const p_bind_buf *u = &D.ubo[gst][slot - 3];
            if (!u->addr || (uint64_t)dw * 4 + 4 > u->size) return 0;
            pend_hazard(u->addr + (uint64_t)dw * 4, 4);
            memcpy(&st[sp - 1], (const uint8_t *)(uintptr_t)u->addr + (uint64_t)dw * 4, 4);
        } else {
            if (sp < 2) return 0;
            uint32_t b = st[--sp], a = st[sp - 1], r;
            switch (o) { case 3: r = a + b; break; case 4: r = a - b; break; case 5: r = a * b; break; case 6: r = a & b; break; case 7: r = a | b; break; case 8: r = a ^ b; break;
                         case 9: r = a << (b & 31); break; case 10: r = a >> (b & 31); break; case 11: r = (uint32_t)((int32_t)a >> (b & 31)); break; default: return 0; }
            st[sp - 1] = r;
        }
    }
    if (sp != 1) return 0;
    *mask = st[0]; return 1;
}
static void do_draw(const draw_args *a) {
    state_init();
    const vks_prog *pr = g_gfx_prog;
    if (!pr) { g_skip[SK_NOPROG]++; return; }
    if (pr->st[VKS_TCS] || pr->st[VKS_TES]) { g_skip[SK_TESS]++; return; }
    VkPrimitiveTopology topo = topo_of(a->prim);
    static int gs_on = -1; if (gs_on < 0) { const char *e = getenv("DM_VK_GEOMETRY"); gs_on = !(e && *e == '0'); if (!gs_on) LOG("[vk] DM_VK_GEOMETRY=0: draws with a geometry stage are skipped"); }
    if (pr->st[VKS_GS] && (!gs_on || g_ab_skip_gs)) { g_skip[SK_TESS]++; return; }
    if (topo == VK_PRIMITIVE_TOPOLOGY_MAX_ENUM || topo == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) { g_skip[SK_PRIM]++; return; }
    if (!a->count) return;
    if (!ring_init() || !dummies_init()) return;
    if (!cb_begin()) return;
    // descriptors first: a texture's first use ends an open pass
    // (the sets are allocated and written before the pass begins and bound inside it)
    if (!pass_begin()) { g_skip[SK_NORT]++; return; }
    pkey k; memset(&k, 0, sizeof k);
    if (!g_pass_fs && pr->st[VKS_FS]) g_pass_fs = pr->st[VKS_FS]->key;
    k.prog = pr->serial; memcpy(k.cf, P.cf, sizeof k.cf); k.df = P.df; k.ncol = P.ncol; k.samples = P.samples;
    k.topo = topo;
    int ll = atomic_load(&g_origin_ll);
    k.front_ccw = ((D.poly.front == NVN_FRONT_FACE_CCW) ^ ll ^ g_queue_origin_ll) != 0;
    static int flipf = -1; if (flipf < 0) { const char *e = getenv("DM_VK_FRONT_FLIP"); flipf = e && *e == '1'; }
    k.front_ccw ^= flipf;
    k.depth_gl = !atomic_load(&g_depth_zero) && VK.depth_clip_control;
    k.rast_discard = D.rast_discard; k.depth_clamp = D.depth_clamp && VK.f_depth_clamp; k.prim_restart = D.prim_restart;
    memcpy(k.blend, D.blend, sizeof k.blend); k.blend_en = D.color.blend_en; k.chmask = D.chmask.mask; k.ds = D.ds; k.poly = D.poly; k.a2c = D.ms.a2c;
    // NVN's alpha test (as OpenGL, none when colour 0 goes to an integer target): the pipeline turns the shader's test on, the support buffer holds function and reference
    g_at_func = D.color.alpha_func && D.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS && !(P.cf[0] && fmt_int(P.cf[0])) ? D.color.alpha_func : 0;
    k.atest = g_at_func != 0; g_at_ref = D.alpha_ref;
    // A rewritten passthrough geometry shader (tools/spv_pack.py marks it) only copies each triangle to the viewports its mask picks; geometry shaders are slow on
    // Adreno (they took half the GPU time of a Mario Tennis Fever frame), so the draw goes without it, once per viewport: a triangle the mask left out of a viewport
    // is cut by that viewport's scissor in the cascades it was made for. DM_VK_GS_LOOP=0: through the geometry shader
    static int gsl = -1; if (gsl < 0) { const char *e = getenv("DM_VK_GS_LOOP"); gsl = !(e && *e == '0'); const char *n = getenv("DM_SPV_VIEWPORTS"); if (n && atoi(n) >= 1 && atoi(n) <= 16) g_gs_vps = (uint32_t)atoi(n); }
    uint32_t gmask = 0;
    k.nogs = pr->st[VKS_GS] && pr->st[VKS_GS]->ngsm && gsl && gs_mask(pr->st[VKS_GS], &gmask);
    if (k.nogs) { static int n; if (n++ < 8) LOG("[vk] a passthrough geometry shader's draw goes without it: viewport mask %#x (shader %016llx)", gmask, (unsigned long long)pr->st[VKS_GS]->key); }
    else if (pr->st[VKS_GS] && pr->st[VKS_GS]->local[0] == 1) { static int n; if (n++ < 4) LOG("[vk] a passthrough geometry shader's mask could not be worked out here (shader %016llx, program %u words): through the geometry shader", (unsigned long long)pr->st[VKS_GS]->key, pr->st[VKS_GS]->ngsm); }
    if (k.nogs && !(gmask & ((1u << g_gs_vps) - 1))) { g_drawn++; return; }   // in no viewport
    if (k.atest) { static int n; if (n++ < 10) LOG("[vk] alpha test: function %u reference %.3f, fragment shader %016llx", g_at_func, g_at_ref, pr->st[VKS_FS] ? (unsigned long long)pr->st[VKS_FS]->key : 0ull); }
    for (int i = 0; i < 8; i++) k.blend[i].target = 0;
    // vertex inputs: each location the vertex shader reads gets NVN attribute <location>, or the zero buffer
    const vks_shader *vs = pr->st[VKS_VS];
    uint64_t vb_off[17] = {0};
    for (int i = 0; vs && i < vs->nvin && k.nva < 32; i++) {
        uint8_t loc = vs->vin[2 * i], ct = vs->vin[2 * i + 1];
        va_st at; memset(&at, 0, sizeof at); if (loc < D.nattr) memcpy(&at, &D.vattr[loc], 4);
        VkFormat f = at.fmt ? vfmt(at.fmt) : VK_FORMAT_UNDEFINED;
        static VkFormat okf[256]; static uint8_t okc[256];   // format support, asked once per format
        if (f && f < 256 && !okc[f]) { okc[f] = 1; okf[f] = vfmt_ok(f) ? f : 0; if (!okf[f]) LOG("[RED] [vk] vertex format %d (NVN %#x) is not supported by this GPU: those attributes read zero", f, at.fmt); }
        if (f && f < 256) f = okf[f];
        if (f && at.stream < 16 && D.vbuf[at.stream].address) {
            k.va[k.nva++] = (typeof(k.va[0])){loc, (uint8_t)at.stream, 1, 0, (uint32_t)f, at.offset};
            k.vb_used |= 1u << at.stream;
            k.vb[at.stream].stride = at.stream < (unsigned)D.nstream ? D.stream[at.stream].stride : 0;
            k.vb[at.stream].rate = at.stream < (unsigned)D.nstream && D.stream[at.stream].divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
        } else k.va[k.nva++] = (typeof(k.va[0])){loc, 0xff, 1, 0, (uint32_t)(ct == 1 ? VK_FORMAT_R32G32B32A32_UINT : ct == 2 ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT), 0};
    }
    // inputs with no attribute read the zero buffer, bound at a stream slot no attribute uses (a GPU need not have more than 16 bindings)
    int zb = -1;
    for (uint32_t i = 0; i < k.nva; i++) if (k.va[i].binding == 0xff) {
        if (zb < 0) { for (int b = 15; b >= 0 && zb < 0; b--) if (!(k.vb_used >> b & 1)) zb = b; if (zb < 0) { g_skip[SK_DESC]++; return; } k.vb_used |= 1u << zb; k.vb[zb].stride = 0; k.vb[zb].rate = VK_VERTEX_INPUT_RATE_VERTEX; }
        k.va[i].binding = (uint8_t)zb;
    }
    // the vertex streams' bytes: from each buffer's bound address to its end (the attributes' offsets are within it)
    for (int b = 0; b < 16; b++) if ((k.vb_used >> b & 1) && b != zb) {
        uint64_t off = ring_put(D.vbuf[b].address, D.vbuf[b].size ? D.vbuf[b].size : 16);
        if (off == ~0ull) { g_skip[SK_RING]++; return; }
        vb_off[b] = off;
    }
    // indices (8-bit ones widened to 16: Vulkan needs an extension for those; quads made into triangles)
    uint64_t ib_off = 0; VkIndexType it = VK_INDEX_TYPE_UINT16;
    int quads = a->prim == NVN_DRAW_PRIMITIVE_QUADS || a->prim == NVN_DRAW_PRIMITIVE_QUAD_STRIP;
    uint32_t icount = a->count; int indexed = a->indexed;
    if (quads) {
        if (a->indexed && !a->ibuf) return;
        int q = quad_indices(a->prim, a, &ib_off, &icount);
        if (q < 0) { g_skip[SK_RING]++; return; }
        if (!q) return;
        it = VK_INDEX_TYPE_UINT32; indexed = 1;
    } else if (a->indexed) {
        uint32_t isz = a->itype == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : a->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 1;
        if (!a->ibuf) return;
        if (isz == 1) {
            ib_off = ring_alloc((uint64_t)a->count * 2); if (ib_off == ~0ull) { g_skip[SK_RING]++; return; }
            const uint8_t *src = (const uint8_t *)(uintptr_t)a->ibuf; uint16_t *dst = (uint16_t *)(RG.p + ib_off); pend_hazard(a->ibuf, a->count);
            for (uint32_t i = 0; i < a->count; i++) dst[i] = src[i] == 0xff && D.prim_restart ? 0xffff : src[i];
        } else {
            ib_off = ring_put(a->ibuf, (uint64_t)a->count * isz); if (ib_off == ~0ull) { g_skip[SK_RING]++; return; }
            it = isz == 4 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        }
    }
    VkPipeline pipe = pipe_get(pr, &k);
    if (!pipe) { g_skip[SK_PIPE]++; return; }
    // descriptors (prep_tex may end the pass for a texture's first use: it is begun again below)
    int r = bind_sets(pr, VK_PIPELINE_BIND_POINT_GRAPHICS);
    if (r <= 0) { g_skip[r < 0 ? SK_RING : SK_DESC]++; return; }
    if (!P.open) {
        if (!pass_begin()) { g_skip[SK_NORT]++; return; }
        // the pass was begun again with the same targets: the sets bound above are still bound (binding state is command-buffer state, not render-pass state)
    }
    if (BS.pipe != pipe) { vkCmdBindPipeline_(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); BS.pipe = pipe; } else BS.saved++;
    for (uint32_t b = 0; b < 17; b++) if (k.vb_used >> b & 1) {
        // vkCmdBindVertexBuffers binds a contiguous range of bindings: bind each on its own
        VkBuffer vb = (int)b == zb ? DZ.zbuf : RG.b; VkDeviceSize vo = (int)b == zb ? 0 : vb_off[b];
        if ((BS.vset >> b & 1) && BS.vb[b] == vb && BS.vo[b] == vo) { BS.saved++; continue; }
        vkCmdBindVertexBuffers_(g_cb, b, 1, &vb, &vo); BS.vb[b] = vb; BS.vo[b] = vo; BS.vset |= 1u << b;
    }
    if (indexed) { if (!BS.ib || BS.io != ib_off || BS.it != it) { vkCmdBindIndexBuffer_(g_cb, RG.b, ib_off, it); BS.ib = 1; BS.io = ib_off; BS.it = it; } else BS.saved++; }
    // the viewports (NVN x, y, w, h; depth range from SetDepthRange); upper-left origin: negative height (see the top of this file). One unless a geometry stage picks one
    static int flipy = -1; if (flipy < 0) { const char *e = getenv("DM_VK_FLIP_Y"); flipy = e && *e ? atoi(e) : -1; if (flipy < 0) flipy = 2; }
    int flip = flipy == 2 ? !ll : flipy;
    uint32_t nvp = pr->st[VKS_GS] && VK.limits.maxViewports > 1 ? (VK.limits.maxViewports < 16 ? VK.limits.maxViewports : 16) : 1;
    if (k.nogs) nvp = g_gs_vps;   // drawn once per viewport below
    VkViewport vps[16]; VkRect2D scs[16];
    int32_t sx = 0, sy = 0; uint32_t sw = P.w, sh = P.h; float ps = P.sc > 0 ? P.sc : 1.0f;   // ps: the targets' resolution scale (guest coordinates to image texels)
    if (D.has_sc && D.sc.w >= 0 && D.sc.h >= 0) {
        int64_t x0 = D.sc.x > 0 ? D.sc.x : 0, y0 = D.sc.y > 0 ? D.sc.y : 0, x1 = (int64_t)D.sc.x + D.sc.w, y1 = (int64_t)D.sc.y + D.sc.h;
        if (ps != 1.0f) { x0 = (int64_t)(x0 * ps); y0 = (int64_t)(y0 * ps); x1 = (int64_t)(x1 * ps + 0.999f); y1 = (int64_t)(y1 * ps + 0.999f); }
        if (x1 > P.w) x1 = P.w;
        if (y1 > P.h) y1 = P.h;
        sx = (int32_t)x0; sy = (int32_t)y0; sw = x1 > x0 ? (uint32_t)(x1 - x0) : 0; sh = y1 > y0 ? (uint32_t)(y1 - y0) : 0;
    }
    for (uint32_t i = 0; i < nvp; i++) {
        float vx = D.vp[i][0] * ps, vy = D.vp[i][1] * ps, vw = D.vp[i][2] * ps, vh = D.vp[i][3] * ps;
        if (vw <= 0 || vh <= 0) { vx = 0; vy = 0; vw = (float)P.w; vh = (float)P.h; }
        float mw = (float)VK.limits.maxViewportDimensions[0], mh = (float)VK.limits.maxViewportDimensions[1];
        if (vw > mw) { vx += (vw - mw) / 2; vw = mw; }   // larger than the GPU allows (Vulkan refuses it): centred at the largest size
        if (vh > mh) { vy += (vh - mh) / 2; vh = mh; }
        vps[i] = (VkViewport){vx, flip ? vy + vh : vy, vw, flip ? -vh : vh, D.dr[0], D.dr[1]};
        scs[i] = (VkRect2D){{sx, sy}, {sw, sh}};   // NVN's other scissors are not recorded (hle_nvn.c SetScissors keeps index 0)
    }
    if (BS.nvp != nvp || memcmp(BS.vp, vps, nvp * sizeof *vps) || memcmp(BS.sc, scs, nvp * sizeof *scs)) {
        vkCmdSetViewport_(g_cb, 0, nvp, vps); vkCmdSetScissor_(g_cb, 0, nvp, scs);
        BS.nvp = nvp; memcpy(BS.vp, vps, nvp * sizeof *vps); memcpy(BS.sc, scs, nvp * sizeof *scs);
    } else BS.saved++;
    if (g_dump_subs && g_dump_lines < 4000) { g_dump_lines++;
        LOG("[vk] dump:   pass %d draw: vertex %016llx fragment %016llx, %u vertices × %u (prim %d), viewport %.0f,%.0f %.0fx%.0f (asked %.0f,%.0f %.0fx%.0f), scissor %d,%d %ux%u, colour mask %#x, blend %#x (target 0: %u %u %u / %u %u %u), alpha test %u, depth test %d write %d",
            g_dump_pass, pr->st[VKS_VS] ? (unsigned long long)pr->st[VKS_VS]->key : 0ull, pr->st[VKS_FS] ? (unsigned long long)pr->st[VKS_FS]->key : 0ull, a->count, a->instances, a->prim,
            vps[0].x, vps[0].y, vps[0].width, vps[0].height, D.vp[0][0], D.vp[0][1], D.vp[0][2], D.vp[0][3], sx, sy, sw, sh, (unsigned)D.chmask.mask, (unsigned)D.color.blend_en, (unsigned)D.blend[0].src_rgb, (unsigned)D.blend[0].dst_rgb, (unsigned)D.blend[0].eq_rgb, (unsigned)D.blend[0].src_a, (unsigned)D.blend[0].dst_a, (unsigned)D.blend[0].eq_a, g_at_func, (int)k.ds.dt, (int)k.ds.dw); }
    vkCmdSetStencilCompareMask_(g_cb, VK_STENCIL_FACE_FRONT_BIT, D.st_vmask[0]); vkCmdSetStencilCompareMask_(g_cb, VK_STENCIL_FACE_BACK_BIT, D.st_vmask[1]);
    vkCmdSetStencilWriteMask_(g_cb, VK_STENCIL_FACE_FRONT_BIT, D.st_wmask[0]); vkCmdSetStencilWriteMask_(g_cb, VK_STENCIL_FACE_BACK_BIT, D.st_wmask[1]);
    vkCmdSetStencilReference_(g_cb, VK_STENCIL_FACE_FRONT_BIT, D.st_ref[0]); vkCmdSetStencilReference_(g_cb, VK_STENCIL_FACE_BACK_BIT, D.st_ref[1]);
    vkCmdSetBlendConstants_(g_cb, D.blend_color);
    vkCmdSetDepthBias_(g_cb, D.po.units, D.po.clamp, D.po.factor);
    uint32_t inst = a->instances ? a->instances : 1;
    for (uint32_t v = 0; v < (k.nogs ? nvp : 1); v++) {
        if (k.nogs && !(gmask >> v & 1)) continue;
        if (k.nogs) { vkCmdSetViewport_(g_cb, 0, 1, &vps[v]); vkCmdSetScissor_(g_cb, 0, 1, &scs[v]); BS.nvp = 0; }
        if (indexed) vkCmdDrawIndexed_(g_cb, icount, inst, 0, quads && !a->indexed ? 0 : a->base_vertex, a->base_instance);
        else vkCmdDraw_(g_cb, a->count, inst, a->first, a->base_instance);
    }
    g_drawn++; g_cb_ops++;
    for (uint32_t c = 0; c < P.ncol && c < 8; c++) if (P.cv[c] && (k.chmask >> (4 * c) & 0xf)) P.cdirty |= 1u << c;
    if (P.dv && ((k.ds.dt && k.ds.dw) || k.ds.st)) P.ddirty = 1;
}
static void draw_op(uint32_t op, const uint8_t *a, uint32_t size) {
    if (op == OP_DRAW) {
        p_draw t; const p_draw *d = draw_rec(a, size, &t);
        if (d->mdi_cnt) pend_hazard(d->mdi_cnt, 4);
        if (d->ind) pend_hazard(d->ind, 20);
        if (d->mdi_cnt && (uint32_t)d->mdi_i >= *(const uint32_t *)(uintptr_t)d->mdi_cnt) return;
        draw_args x = {d->prim, 0, (uint32_t)d->count, (uint32_t)d->instances, (uint32_t)d->first, (uint32_t)d->base_instance, 0, 0, 0};
        if (d->ind) { const uint32_t *g = (const uint32_t *)(uintptr_t)d->ind; x.count = g[0]; x.instances = g[1]; x.first = g[2]; x.base_instance = g[3]; }   // the arguments as the CPU sees them
        do_draw(&x);
    } else if (op == OP_DRAW_IDX) {
        p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, size, &t);
        if (d->mdi_cnt) pend_hazard(d->mdi_cnt, 4);
        if (d->ind) pend_hazard(d->ind, 20);
        if (d->mdi_cnt && (uint32_t)d->mdi_i >= *(const uint32_t *)(uintptr_t)d->mdi_cnt) return;
        draw_args x = {d->prim, 1, (uint32_t)d->count, (uint32_t)d->instances, 0, (uint32_t)d->base_instance, d->base_vertex, d->ibuf, d->itype};
        if (d->ind) {
            const uint32_t *g = (const uint32_t *)(uintptr_t)d->ind; uint32_t isz = d->itype == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 1;
            x.count = g[0]; x.instances = g[1]; x.ibuf = d->ibuf + (uint64_t)g[2] * isz; x.base_vertex = (int32_t)g[3]; x.base_instance = g[4];
        }
        do_draw(&x);
    } else if (op == OP_MULTI_DRAW) {
        const p_multi_draw *m = (const p_multi_draw *)a;
        if (m->cnt) pend_hazard(m->cnt, 4);
        if (m->ind && m->max) pend_hazard(m->ind, (uint64_t)m->max * (m->stride ? (uint64_t)m->stride : 20));
        uint32_t n = (uint32_t)m->max; if (m->cnt) { uint32_t c = *(const uint32_t *)(uintptr_t)m->cnt; if (c < n) n = c; }
        int idx = m->itype >= 0; uint64_t stride = m->stride ? (uint64_t)m->stride : idx ? 20 : 16;
        uint32_t isz = m->itype == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : m->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 1;
        for (uint32_t i = 0; i < n && i < 65536; i++) {
            const uint32_t *g = (const uint32_t *)(uintptr_t)(m->ind + i * stride);
            draw_args x = idx ? (draw_args){m->prim, 1, g[0], g[1], 0, g[4], (int32_t)g[3], m->ibuf + (uint64_t)g[2] * isz, m->itype}
                              : (draw_args){m->prim, 0, g[0], g[1], g[2], g[3], 0, 0, 0};
            do_draw(&x);
        }
    }
}

// ---------------------------------------------------------------- a compute dispatch
typedef struct { uint64_t serial; VkPipeline p; } cpipe;
#define CP_CAP 4096
static cpipe g_cp[CP_CAP];
static void dispatch_op(const p_dispatch *d) {
    const vks_prog *pr = g_cs_prog;
    if (!pr || !pr->st[VKS_CS]) return;
    if (!ring_init() || !dummies_init() || !cb_begin()) return;
    pass_end();
    uint32_t i = (uint32_t)(pr->serial * 0x9E3779B97F4A7C15ull >> 52) & (CP_CAP - 1);
    for (uint32_t n = 0; n < CP_CAP && g_cp[i].serial && g_cp[i].serial != pr->serial; n++) i = (i + 1) & (CP_CAP - 1);
    if (g_cp[i].serial != pr->serial) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, pr->mod[VKS_CS], "main", NULL}, pr->layout, VK_NULL_HANDLE, -1};
        if (!g_pcache) { VkPipelineCacheCreateInfo pc = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, 0, NULL}; vkCreatePipelineCache_(VK.dev, &pc, NULL, &g_pcache); }
        VkPipeline p = VK_NULL_HANDLE;
        if (vkCreateComputePipelines_(VK.dev, g_pcache, 1, &ci, NULL, &p)) { static int n; if (n++ < 10) LOG("[RED] [vk] a compute pipeline was refused: shader %016llx", (unsigned long long)pr->st[VKS_CS]->key); }
        g_cp[i].serial = pr->serial; g_cp[i].p = p; if (p) g_pipes++;
    }
    if (!g_cp[i].p) return;
    if (bind_sets(pr, VK_PIPELINE_BIND_POINT_COMPUTE) <= 0) return;
    pass_end();   // a texture's first use may have needed command-buffer work
    vkCmdBindPipeline_(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cp[i].p);
    uint32_t x = (uint32_t)d->x, y = (uint32_t)d->y, z = (uint32_t)d->z;
    if (d->ind) { pend_hazard(d->ind, 12); const uint32_t *g = (const uint32_t *)(uintptr_t)d->ind; x = g[0]; y = g[1]; z = g[2]; }
    if (x && y && z) { prof_mark(1, x, y, z, 0, 1, 0, 0); vkCmdDispatch_(g_cb, x, y, z); prof_mark(0, x, y, z, 0, 1, pr->st[VKS_CS] ? pr->st[VKS_CS]->key : 0, 1); full_barrier(); g_disp_done++; g_cb_ops++; }
}


// ---------------------------------------------------------------- frame dumps (debugging): every render pass of the next few submissions saved as a picture
// Asked for with L3 + R3 held together, F12, or DM_VK_DUMP_AT=<seconds after the first frame>. Each pass's colour targets and depth are copied out when the pass ends
// and written as .ppm pictures (at most 800 pixels wide) into ~/nxnative-dump (DM_VK_DUMP_DIR), named <pass>_<target>_<format>.ppm, with a list in the log
_Atomic int vkw_dump_req;   // set by the window (gfx_vk_window.c input_poll)
static int g_dump_files; static char g_dump_dir[400];
static uint32_t dump_bpp(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R8_UNORM: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_D16_UNORM: return 2;
    default: return 0;
    }
}
static float half_f(uint16_t h) {
    uint32_t e = (h >> 10) & 31, m = h & 1023; float v;
    if (!e) v = m / 16777216.0f; else if (e == 31) v = m ? 0 : 65504.0f; else v = ldexpf(1.0f + m / 1024.0f, (int)e - 15);
    return h & 0x8000 ? -v : v;
}
static float uf11(uint32_t x, int mb) {   // the small unsigned floats of B10G11R11 (mb mantissa bits, 5 exponent bits)
    uint32_t e = x >> mb, m = x & ((1u << mb) - 1);
    if (!e) return ldexpf((float)m / (1u << mb), -14); if (e == 31) return 65504.0f;
    return ldexpf(1.0f + (float)m / (1u << mb), (int)e - 15);
}
static void dump_texel(VkFormat f, const uint8_t *p, float o[4]) {
    uint32_t u; memcpy(&u, p, 4 <= dump_bpp(f) ? 4 : dump_bpp(f)); const uint16_t *h = (const uint16_t *)p; const float *fl = (const float *)p;
    o[0] = o[1] = o[2] = 0; o[3] = 1;
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32: case VK_FORMAT_R8G8B8A8_UINT:
        for (int i = 0; i < 4; i++) o[i] = p[i] / 255.0f; break;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: o[0] = p[2] / 255.0f; o[1] = p[1] / 255.0f; o[2] = p[0] / 255.0f; o[3] = p[3] / 255.0f; break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: o[0] = (u & 1023) / 1023.0f; o[1] = (u >> 10 & 1023) / 1023.0f; o[2] = (u >> 20 & 1023) / 1023.0f; break;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: o[2] = (u & 1023) / 1023.0f; o[1] = (u >> 10 & 1023) / 1023.0f; o[0] = (u >> 20 & 1023) / 1023.0f; break;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: o[0] = uf11(u & 2047, 6); o[1] = uf11(u >> 11 & 2047, 6); o[2] = uf11(u >> 22, 5); break;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: { float sc = ldexpf(1, (int)(u >> 27) - 24); o[0] = (u & 511) * sc; o[1] = (u >> 9 & 511) * sc; o[2] = (u >> 18 & 511) * sc; break; }
    case VK_FORMAT_R16G16B16A16_SFLOAT: for (int i = 0; i < 4; i++) o[i] = half_f(h[i]); break;
    case VK_FORMAT_R16G16B16A16_UNORM: for (int i = 0; i < 4; i++) o[i] = h[i] / 65535.0f; break;
    case VK_FORMAT_R16G16_SFLOAT: o[0] = half_f(h[0]); o[1] = half_f(h[1]); break;
    case VK_FORMAT_R16G16_UNORM: o[0] = h[0] / 65535.0f; o[1] = h[1] / 65535.0f; break;
    case VK_FORMAT_R32G32_SFLOAT: o[0] = fl[0]; o[1] = fl[1]; break;
    case VK_FORMAT_R32G32B32A32_SFLOAT: for (int i = 0; i < 4; i++) o[i] = fl[i]; break;
    case VK_FORMAT_R8_UNORM: o[0] = o[1] = o[2] = p[0] / 255.0f; break;
    case VK_FORMAT_R8G8_UNORM: o[0] = p[0] / 255.0f; o[1] = p[1] / 255.0f; break;
    case VK_FORMAT_R16_SFLOAT: o[0] = o[1] = o[2] = half_f(h[0]); break;
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_D16_UNORM: o[0] = o[1] = o[2] = h[0] / 65535.0f; break;
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: o[0] = o[1] = o[2] = fl[0]; break;
    case VK_FORMAT_R32_UINT: o[0] = o[1] = o[2] = (float)u; break;
    case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: o[0] = o[1] = o[2] = (u & 0xffffff) / 16777215.0f; break;
    default: break;
    }
}
// Written after the submission completed: scaled to at most 800 pixels wide. Colours beyond 0..1 (HDR) are squeezed (x / (1 + x)) when any texel is above 1;
// one-channel images (depth, masks) are stretched from their smallest to their largest value, so a nearly flat depth buffer still shows its shapes
static void dump_write(const stage_t *s) {
    uint32_t bpp = dump_bpp((VkFormat)s->dump_fmt), w = s->dw, h = s->dh, step = (w + 799) / 800, ow, oh;
    if (step < 1) step = 1;
    ow = (w + step - 1) / step; oh = (h + step - 1) / step;
    float mx = 0, mn = 1e30f, hi = -1e30f; int mono = 0; VkFormat f = (VkFormat)s->dump_fmt;
    switch (f) { case VK_FORMAT_R8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_D16_UNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT:
                 case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: mono = 1; break; default: break; }
    for (uint32_t y = 0; y < h; y += step) for (uint32_t x = 0; x < w; x += step) {
        float o[4]; dump_texel(f, (const uint8_t *)s->p + ((uint64_t)y * w + x) * bpp, o);
        for (int c = 0; c < 3; c++) { if (o[c] > mx) mx = o[c]; if (o[c] < mn) mn = o[c]; if (o[c] > hi) hi = o[c]; }
    }
    FILE *fp = fopen(s->dump_name, "wb"); if (!fp) { LOG("[vk] dump: cannot write %s", s->dump_name); return; }
    fprintf(fp, "P6\n%u %u\n255\n", ow, oh);
    uint8_t *row = malloc(ow * 3);
    for (uint32_t y = 0; y < h; y += step) {
        uint32_t ox = 0;
        for (uint32_t x = 0; x < w; x += step, ox++) {
            float o[4]; dump_texel(f, (const uint8_t *)s->p + ((uint64_t)y * w + x) * bpp, o);
            for (int c = 0; c < 3; c++) {
                float v = o[c];
                if (mono) v = hi > mn ? (v - mn) / (hi - mn) : 0;
                else if (mx > 1.0f) v = v > 0 ? v / (1 + v) : 0;
                row[ox * 3 + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + 0.5f);
            }
        }
        fwrite(row, 1, ow * 3, fp);
    }
    free(row); fclose(fp); g_dump_files++;
    LOG("[vk] dump: %s (%ux%u, values %.3g .. %.3g)", s->dump_name, w, h, mn, mx);
}
static void dump_tex(vtex *v, uint32_t lv, uint32_t l0, const char *tag) {
    uint32_t bpp = dump_bpp(v->fmt);
    if (!bpp || v->samples > 1 || !v->img) { LOG("[vk] dump: pass %d %s: serial %u format %d, %u samples: not saved (format or multisampling)", g_dump_pass, tag, (unsigned)v->serial, v->fmt, v->samples); return; }
    uint32_t w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1;
    stage_t *st = stage_new((uint64_t)w * h * bpp); if (!st) return;
    st->dump_fmt = v->fmt; st->dw = w; st->dh = h;
    snprintf(st->dump_name, sizeof st->dump_name, "%s/%03d_%s_s%u_f%d.ppm", g_dump_dir, g_dump_pass, tag, (unsigned)v->serial, v->fmt);
    VkImageAspectFlags a = fmt_aspects(v->fmt); a = a & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    VkBufferImageCopy bic = {0, 0, 0, {a, lv, v->is3d ? 0 : l0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, st->b, 1, &bic);
}
static void dump_pass(void) {
    if (!g_dump_subs) return;
    if (g_dump_pass >= 600) return;
    int pics = g_dump_light ? P.w >= 1280 && P.ncol : g_dump_pass < 40 || P.w >= 256;   // light dumps (frequent ones): the full-size colour passes only
    LOG("[vk] dump: pass %d: %llu draws, %ux%u, %u colour targets%s", g_dump_pass, (unsigned long long)(g_drawn - P.draws0), P.w, P.h, P.ncol, P.dv ? " + depth" : "");
    char tag[16];
    for (uint32_t i = 0; pics && i < P.ncol && i < 8; i++) if (P.cv[i]) { snprintf(tag, sizeof tag, "c%u", i); dump_tex(P.cv[i], P.clv[i], P.cl0[i], tag); }
    if (pics && P.dv) dump_tex(P.dv, P.dlv, P.dl0, "depth");
    for (int q = 0; q < g_ndump_tex; q++) {   // the textures its draws sampled (those dump_bpp knows: not block-compressed ones)
        if (g_ndump_saved < 512) g_dump_saved[g_ndump_saved++] = g_dump_tex[q]->serial;
        if (dump_bpp(g_dump_tex[q]->fmt)) dump_tex(g_dump_tex[q], 0, 0, "tex");
    }
    g_ndump_tex = 0;
    for (uint32_t i = 0; i < P.ncol && i < 8; i++) if (P.cv[i]) LOG("[vk] dump:   pass %d target c%u: texture serial %llu, drawn as format %d into the image of serial %llu %ux%u format %d", g_dump_pass, i, (unsigned long long)P.cser[i], P.cvf[i], (unsigned long long)P.cv[i]->serial, P.cv[i]->w, P.cv[i]->h, P.cv[i]->fmt);
    full_barrier();
    g_dump_pass++;
}
// Each submission's end: a request starts a dump of the next 8 submissions
static void dump_submitted(void) {
    if (g_dump_subs) {
        if (!--g_dump_subs) {   // complete: it takes the place of the last one (a game closed during a dump keeps the one before)
            pend_wait();   // the last submission's pictures are written when it completes
            char fin[400]; size_t l = strlen(g_dump_dir); snprintf(fin, sizeof fin, "%.*s", (int)(l > 4 ? l - 4 : l), g_dump_dir);
            char cmd[900]; snprintf(cmd, sizeof cmd, "rm -rf '%s' && mv '%s' '%s'", fin, g_dump_dir, fin);
            if (system(cmd)) LOG("[vk] dump: could not move %s to %s", g_dump_dir, fin);
            LOG("[vk] dump finished: %d passes, %d pictures in %s", g_dump_pass, g_dump_files, fin);
        }
        return;
    }
    static double at = -2; static uint64_t t0;
    if (at == -2) { const char *e = getenv("DM_VK_DUMP_AT"); at = e && *e ? atof(e) : -1; }
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    if (!t0) t0 = now;
    int timed = at >= 0 && (now - t0) / 1e9 >= at;
    // or a file: `touch ~/nxnative-dump-now` from a terminal while the game runs (checked twice a second)
    static uint64_t last_chk; int file = 0;
    if (now - last_chk > 500000000ull) {
        last_chk = now; const char *home = getenv("HOME"); char fn[400];
        snprintf(fn, sizeof fn, "%s/nxnative-dump-now", home ? home : "/tmp");
        if (!access(fn, F_OK)) { unlink(fn); file = 1; LOG("[vk] frame dump asked for (%s)", fn); }
    }
    // or every 20 seconds while ~/nxnative-dump-auto exists (made before starting the game: each dump replaces the one before, so the last one is of what was on
    // screen just before the game was closed)
    static int auto_on = -1; static uint64_t auto_last;
    if (auto_on < 0) { const char *home = getenv("HOME"); char fn[400]; snprintf(fn, sizeof fn, "%s/nxnative-dump-auto", home ? home : "/tmp"); auto_on = !access(fn, F_OK); if (auto_on) LOG("[vk] %s exists: a frame dump every 20 seconds", fn); }
    static uint64_t every = 20000000000ull;
    if (auto_on == 1) { const char *home = getenv("HOME"); char fn[400]; snprintf(fn, sizeof fn, "%s/nxnative-dump-auto", home ? home : "/tmp");
        FILE *fp = fopen(fn, "r"); double sec = 0; if (fp) { if (fscanf(fp, "%lf", &sec) != 1) sec = 0; fclose(fp); }
        if (sec >= 1) { every = (uint64_t)(sec * 1e9); LOG("[vk] frame dumps every %.0f seconds", sec); } auto_on = 2; }
    int autod = auto_on && now - auto_last > every && now - t0 > every;
    if (autod) auto_last = now;
    if (!atomic_exchange(&vkw_dump_req, 0) && !timed && !file && !autod) return;
    if (timed) at = -1;
    const char *d = getenv("DM_VK_DUMP_DIR"), *home = getenv("HOME");
    if (d && *d) snprintf(g_dump_dir, sizeof g_dump_dir, "%s.new", d); else snprintf(g_dump_dir, sizeof g_dump_dir, "%s/nxnative-dump.new", home ? home : "/tmp");   // renamed when complete
    char cmd[480]; snprintf(cmd, sizeof cmd, "rm -rf '%s' && mkdir -p '%s'", g_dump_dir, g_dump_dir);
    if (system(cmd)) LOG("[vk] dump: could not make %s", g_dump_dir);
    g_dump_light = auto_on && every < 10000000000ull;   // every few seconds: 3 submissions, the full-size pictures only, so the game keeps running
    g_dump_subs = g_dump_light ? 3 : 8; g_dump_pass = 0; g_dump_files = 0; g_dump_lines = 0; g_ndump_tex = 0; g_ndump_saved = 0;
    LOG("[vk] dump: saving the render passes of the next %d submissions into %s%s", g_dump_subs, g_dump_dir, g_dump_light ? " (light)" : "");
}

// ---------------------------------------------------------------- the submission's end: the ring and descriptor pools start over
// Linear textures the GPU wrote: read back into guest memory with the submission (the CPU may read or keep writing them), then protected again
static uint64_t g_lin_rb, g_lin_rb_bytes;
static void linear_readbacks(void) {
    for (int i = 0; i < g_nlw; i++) {
        vtex *v = g_lw[i]; if (!v->img || !v->gmem || !v->rowbytes) continue;
        stage_t *st = stage_new((uint64_t)v->rowbytes * v->rows); if (!st) continue;
        st->rb_dst = v->gmem; st->lin_rb = 1; st->rb_row = v->stride; st->rb_src_row = v->rowbytes; st->rb_rows = v->rows; st->rb_n = v->rowbytes;
        VkBufferImageCopy bic = {0, 0, 0, {fmt_aspects(v->fmt) & (VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT), 0, 0, 1}, {0, 0, 0}, {v->w, v->h, 1}};
        if (!bic.imageSubresource.aspectMask) bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        vkCmdCopyImageToBuffer_(g_cb, v->img, VK_IMAGE_LAYOUT_GENERAL, st->b, 1, &bic);
        g_lin_rb++; g_lin_rb_bytes += (uint64_t)v->rowbytes * v->rows;
    }
    if (g_nlw) full_barrier();
}
static void rec_reset(void) {   // a new submission starts recording: the ring half and the descriptor pools it was given are its own from the start
    RG.used = RG.base; DZ.support_off = 0; g_support_n++;
    if (g_nrh_used) { memset(g_rh, 0, sizeof g_rh); g_nrh_used = 0; }
    g_dp_cur = 0; dsc_reset();
}
static void draw_summary(char *out, size_t n) {
    int o = snprintf(out, n, "drawn %llu (%llu of them quads), dispatched %llu, pipelines %llu, textures uploaded %llu (%.0f MB, %llu failed, %llu again after CPU writes, %llu untracked; linear textures read back %llu times, %.0f MB)", (unsigned long long)g_drawn, (unsigned long long)g_quads, (unsigned long long)g_disp_done,
                     (unsigned long long)g_pipes, (unsigned long long)g_up_tex, g_up_bytes / 1048576.0, (unsigned long long)g_up_fail, (unsigned long long)g_reup, (unsigned long long)g_untracked, (unsigned long long)g_lin_rb, g_lin_rb_bytes / 1048576.0);
    for (int i = 0; i < SK_N && o < (int)n; i++) if (g_skip[i]) o += snprintf(out + o, n - o, "; skipped (%s) %llu", k_sk[i], (unsigned long long)g_skip[i]);
    if (g_tex_missing && o < (int)n) o += snprintf(out + o, n - o, "; texture bindings without a usable view %llu", (unsigned long long)g_tex_missing);
    if (g_storage_unsup && o < (int)n) o += snprintf(out + o, n - o, "; shader images in a format that cannot be written %llu", (unsigned long long)g_storage_unsup);
    if (g_tex_unreg && o < (int)n) o += snprintf(out + o, n - o, "; bound texture IDs not registered %llu", (unsigned long long)g_tex_unreg);
    if (g_ring_forgot && o < (int)n) o += snprintf(out + o, n - o, "; copies made again after the command stream changed their bytes %llu", (unsigned long long)g_ring_forgot);
    if (g_fb_split && o < (int)n) o += snprintf(out + o, n - o, "; passes ended for a draw reading its own target %llu", (unsigned long long)g_fb_split);
    if ((g_clear_load || g_clear_att || g_clear_img) && o < (int)n) o += snprintf(out + o, n - o, "; render target clears: as a pass's load %llu, inside a pass %llu, as image clears %llu (%llu not attachments of the next pass; passes split for a cleared target read as a texture %llu", (unsigned long long)g_clear_load, (unsigned long long)g_clear_att, (unsigned long long)g_clear_img, (unsigned long long)g_clear_untaken, (unsigned long long)g_clear_split);
    for (int r = 0; r < 4 && o < (int)n; r++) { int b = -1; for (int i = 0; i < 128; i++) if (g_clear_why[i] && (b < 0 || g_clear_why[i] > g_clear_why[b])) b = i; if (b < 0) break;
        o += snprintf(out + o, n - o, ", %llu before op %d", (unsigned long long)g_clear_why[b], b); g_clear_why[b] = 0; }
    if (o < (int)n) o += snprintf(out + o, n - o, ")");
    if ((g_tbuf_n || g_tbuf_miss) && o < (int)n) o += snprintf(out + o, n - o, "; texture buffers bound %llu, unusable %llu", (unsigned long long)g_tbuf_n, (unsigned long long)g_tbuf_miss);
    if (g_tex_unres && o < (int)n) o += snprintf(out + o, n - o, "; texture handles not decoded %llu", (unsigned long long)g_tex_unres);
    if (g_unsup_sbuf_ubo && o < (int)n) snprintf(out + o, n - o, "; storage buffers addressed through uniforms (not bound) %llu", (unsigned long long)g_unsup_sbuf_ubo);
}
