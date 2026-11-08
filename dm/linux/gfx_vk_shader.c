// Stage C of the Vulkan graphics layer (dm/linux/GRAPHICS.md), shaders: the game's shaders translated to SPIR-V ahead of time (scripts/build_linux.py --shaders: shader_split.py, then
// dm/linux/tools/spv_build.py, which runs the Mac build's translation unchanged, then dm/linux/tools/spv_pack.py), looked up by the key the NVN layer computes for each program stage
// (hle_nvn.c prog_key: a hash of the control section and the code, the same key the Mac shader libraries use).
//
// The translator puts every binding of a shader in descriptor set 0, each stage numbering its own from 0 (binding 0: its support buffer). In a pipeline the stages' sets must not
// collide, so each stage's module is made with its DescriptorSet decorations rewritten to its own set (vertex 0, tessellation control 1, evaluation 2, geometry 3, fragment 4; compute 0).
#include "dm.h"
#include "../gfx/nvn/nvn_obj.h"
#include "gfx_vk.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const uint8_t *g_pack; static size_t g_pack_len; static uint32_t g_n;
static int g_init;   // 0 not tried, 1 open, -1 none
static uint64_t g_progs, g_missing, g_failed;

int vks_init(void) {
    if (g_init) return g_init > 0;
    g_init = -1;
    const char *path = getenv("DM_SHADERS");
    if (!path || !*path) { LOG("[YELLOW] [vk] no shader pack (DM_SHADERS): draws are skipped. Build with: scripts/build_linux.py <game> --shaders"); return 0; }
    int fd = open(path, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size < 16) { LOG("[RED] [vk] shader pack %s: %s", path, strerror(errno)); if (fd >= 0) close(fd); return 0; }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) { LOG("[RED] [vk] shader pack %s: mmap: %s", path, strerror(errno)); return 0; }
    uint32_t ver, n; memcpy(&ver, (uint8_t *)m + 4, 4); memcpy(&n, (uint8_t *)m + 8, 4);
    if (memcmp(m, "DMSP", 4) || ver != 2 || 16 + (uint64_t)n * 16 > (uint64_t)st.st_size) { LOG("[RED] [vk] %s is not a shader pack this host reads (version %u; build again with --shaders to repack)", path, ver); munmap(m, (size_t)st.st_size); return 0; }
    g_pack = m; g_pack_len = (size_t)st.st_size; g_n = n; g_init = 1;
    LOG("[vk] shader pack: %u shaders (%.1f MB) from %s", n, st.st_size / 1048576.0, path);
    return 1;
}

// The pack's entry for a key (binary search of the sorted index); the result points into the mapped pack
typedef struct { uint64_t key; uint32_t off, len; } pack_idx;
static int entry_of(uint64_t key, vks_shader *s) {
    uint32_t lo = 0, hi = g_n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2; pack_idx e; memcpy(&e, g_pack + 16 + (size_t)mid * 16, sizeof e);
        if (e.key == key) {
            if ((uint64_t)e.off + e.len > g_pack_len || e.len < 12) return 0;
            const uint8_t *p = g_pack + e.off;
            s->gsm = NULL; s->ngsm = 0;   // (the caller's struct is not zeroed: a shader without a mask program has none)
            s->key = key; s->model = p[0]; s->nb = p[1]; s->nvin = p[2] & 0x7f; s->c1_pad = p[2] >> 7; s->fout = p[3]; memcpy(s->local, p + 4, 6); memcpy(&s->c1_len, p + 10, 2);
            s->b = (const vks_bind *)(p + 12); s->vin = p + 12 + sizeof(vks_bind) * s->nb; s->c1 = s->vin + 2 * s->nvin;
            uint32_t h = 12 + sizeof(vks_bind) * s->nb + 2 * s->nvin + s->c1_len; h += -h & 3;
            if (s->model == 3 && s->local[0] == 1 && s->local[1]) { s->gsm = (const uint32_t *)(p + h); s->ngsm = s->local[1]; h += 4 * s->ngsm; }   // the mask program after c1
            if (h >= e.len) return 0;
            s->spv = (const uint32_t *)(p + h); s->nspv = (e.len - h) / 4;
            memset(s->xfb, 0, sizeof s->xfb);
            if (s->model == 0) {   // transform feedback (tools/spv_xfb.py): each capture buffer's stride, from its outputs' XfbBuffer and XfbStride decorations
                uint32_t tgt[64], buf[64], nt = 0;
                for (uint32_t i = 5; i < s->nspv && nt < 64; ) {
                    uint32_t n = s->spv[i] >> 16, op = s->spv[i] & 0xffff; if (!n) break;
                    if (op == 54) break;   // OpFunction: past the decorations
                    if (op == 71 && n == 4 && s->spv[i + 2] == 36) { tgt[nt] = s->spv[i + 1]; buf[nt++] = s->spv[i + 3]; }
                    if (op == 71 && n == 4 && s->spv[i + 2] == 37) for (uint32_t k = 0; k < nt; k++) if (tgt[k] == s->spv[i + 1] && buf[k] < 4) s->xfb[buf[k]] = s->spv[i + 3];
                    i += n;
                }
            }
            return s->spv[0] == 0x07230203;
        }
        if (e.key < key) lo = mid + 1; else hi = mid;
    }
    return 0;
}

// Found shaders, kept (open addressing by key; each lives as long as the process)
#define SH_CAP (1u << 15)
static vks_shader *g_sh[SH_CAP];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static const vks_shader *find_locked(uint64_t key) {
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 49) & (SH_CAP - 1);
    for (uint32_t k = 0; k < SH_CAP; k++, i = (i + 1) & (SH_CAP - 1)) {
        if (!g_sh[i]) {
            vks_shader s; if (!entry_of(key, &s)) return NULL;
            g_sh[i] = malloc(sizeof s); *g_sh[i] = s; return g_sh[i];
        }
        if (g_sh[i]->key == key) return g_sh[i];
    }
    return NULL;
}
const vks_shader *vks_find(uint64_t key) {
    if (!vks_init()) return NULL;
    pthread_mutex_lock(&g_mu); const vks_shader *s = find_locked(key); pthread_mutex_unlock(&g_mu);
    return s;
}

// A module with every DescriptorSet decoration set to set
// Modules are shared: one per shader and descriptor set, whatever programs use it. Made per program, a vertex shader in 100 programs was 100 modules, each holding its own copy of the
// SPIR-V in the driver (and in Steam's Fossilize layer); in Ocarina of Time's first scene, with shaders of 400 KB and more, memory and swap ran out
static VkShaderModule module_make(const vks_shader *s, uint32_t set);
static VkShaderModule module_for(const vks_shader *s, uint32_t set) {
    enum { MC_CAP = 1 << 15 };
    static struct { const vks_shader *s; uint32_t set; VkShaderModule m; } *mc; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (!mc) mc = calloc(MC_CAP, sizeof *mc);
    uint32_t i = (uint32_t)(((uintptr_t)s * 0x9E3779B97F4A7C15ull + set) >> 49) & (MC_CAP - 1);
    for (uint32_t n = 0; n < MC_CAP && mc[i].s; n++, i = (i + 1) & (MC_CAP - 1))
        if (mc[i].s == s && mc[i].set == set) { VkShaderModule m = mc[i].m; pthread_mutex_unlock(&mu); return m; }
    VkShaderModule m = module_make(s, set);
    if (m && !mc[i].s) { mc[i].s = s; mc[i].set = set; mc[i].m = m; }
    pthread_mutex_unlock(&mu);
    return m;
}
static VkShaderModule module_make(const vks_shader *s, uint32_t set) {
    uint32_t *w = malloc(s->nspv * 4); memcpy(w, s->spv, s->nspv * 4);
    for (uint32_t i = 5; i < s->nspv; ) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (!n) break;
        if (op == 71 && n >= 4 && w[i + 2] == 34) w[i + 3] = set;   // OpDecorate <id> DescriptorSet <set>
        i += n;
    }
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, s->nspv * 4, w};
    VkShaderModule m = VK_NULL_HANDLE; VkResult r = vkCreateShaderModule_(VK.dev, &ci, NULL, &m);
    free(w);
    if (r) { LOG("[RED] [vk] shader %016llx: vkCreateShaderModule failed (%d)", (unsigned long long)s->key, r); return VK_NULL_HANDLE; }
    return m;
}
static const VkShaderStageFlags k_stage[6] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                                               VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT};
static VkDescriptorSetLayout set_layout(const vks_shader *s, int dyn) {   // s null: an empty set (a stage the program does not have, below one it has); dyn: uniform buffers dynamic
    VkDescriptorSetLayoutBinding b[256]; uint32_t n = 0;
    for (int i = 0; s && i < s->nb && n < 256; i++) {
        VkDescriptorType t = (VkDescriptorType)s->b[i].vktype; if (dyn && t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) t = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        b[n++] = (VkDescriptorSetLayoutBinding){s->b[i].binding, t, s->b[i].count ? s->b[i].count : 1, k_stage[s->model], NULL};
    }
    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, b};   // sets come from per-submission pools (a shader can have more bindings than the push limit)
    VkDescriptorSetLayout l = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout_(VK.dev, &ci, NULL, &l)) LOG("[RED] [vk] shader %016llx: its descriptor set layout was refused", s ? (unsigned long long)s->key : 0ull);
    return l;
}

// Programs, by serial (a program set again gets a new serial: hle_nvn.c nvnProgramSetShaders)
#define PR_CAP (1u << 14)
static vks_prog *g_pr[PR_CAP];
static int g_logged_missing;
// A program's shaders, modules, set layouts and pipeline layout. gs_class >= 0: its geometry shader is the translator's ordinary one for that input topology
// (spv_pack.py packs <key>_g<t> under key ^ GVAR * (t + 1)), not the base translation, which uses NVIDIA's passthrough geometry shaders
#define GVAR 0x9E3779B97F4A7C15ull
static vks_prog *prog_build(const prog_t *p, uint64_t serial, int gs_class) {
    vks_prog *r = calloc(1, sizeof *r); r->serial = serial; r->ok = 1; r->gs_class = gs_class; r->prog = p;
    for (unsigned st = 0; st < p->nstage && st < 6; st++) {
        uint8_t kb[8]; if (!prog_key(p, st, kb)) continue;
        uint64_t key = 0; for (int j = 0; j < 8; j++) key = key << 8 | kb[j];
        const vks_shader *s = find_locked(key);
        if (!s) {
            r->ok = 0; g_missing++;
            if (g_logged_missing++ < 20) LOG("[RED] [vk] program stage %u: shader %016llx is not in the pack (its draws are skipped)", st, (unsigned long long)key);
            continue;
        }
        if (s->model > 5) { r->ok = 0; continue; }
        if (s->model == VKS_GS) {
            r->has_gs = 1;
            if (gs_class >= 0) {
                const vks_shader *v = find_locked(key ^ (GVAR * (uint64_t)(gs_class + 1)));
                if (!v) { r->ok = 0; static int n; if (n++ < 10) LOG("[RED] [vk] geometry shader %016llx has no translation for input topology %d (build again with --shaders)", (unsigned long long)key, gs_class); continue; }
                s = v;
            }
        }
        if (s->model == VKS_CS) r->compute = 1;
        r->st[s->model] = s;
    }
    if (r->ok) {
        int nset = 1;
        for (int m = 0; m < 6; m++) if (r->st[m] && vks_set_of(m) + 1 > nset) nset = vks_set_of(m) + 1;
        if (r->compute) nset = 1;
        r->nset = nset;
        for (int m = 0; m < 6; m++) if (r->st[m] && (m == VKS_CS) == (r->compute != 0)) { r->mod[m] = module_for(r->st[m], (uint32_t)vks_set_of(m)); if (!r->mod[m]) r->ok = 0; }
        // Uniform buffers as dynamic ones (their ring offsets given at bind time), when the program's count is within the GPU's limit: draws that differ only in
        // their uniform data reuse one descriptor set (gfx_vk_draw.inc bind_sets). DM_VK_DYN_UBO=0: static ones
        static int dyn_on = -1; if (dyn_on < 0) { const char *e = getenv("DM_VK_DYN_UBO"); dyn_on = !(e && *e == '0'); }
        uint32_t nubo = 0;
        for (int m = 0; m < 6; m++) if (r->st[m] && (m == VKS_CS) == (r->compute != 0)) for (int i = 0; i < r->st[m]->nb; i++) if (r->st[m]->b[i].vktype == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) nubo += r->st[m]->b[i].count ? r->st[m]->b[i].count : 1;
        r->dyn = dyn_on && nubo && nubo <= VK.limits.maxDescriptorSetUniformBuffersDynamic;
        for (int k = 0; k < nset && r->ok; k++) {
            const vks_shader *s = NULL;
            for (int m = 0; m < 6; m++) if (r->st[m] && vks_set_of(m) == k && (m == VKS_CS) == (r->compute != 0)) s = r->st[m];
            r->set[k] = set_layout(s, r->dyn); if (!r->set[k]) r->ok = 0;
        }
        if (r->ok && nset > (int)VK.limits.maxBoundDescriptorSets) { r->ok = 0; LOG("[RED] [vk] a program needs %d descriptor sets, the GPU allows %u", nset, VK.limits.maxBoundDescriptorSets); }
        if (r->ok) {
            VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nset, r->set, 0, NULL};
            if (vkCreatePipelineLayout_(VK.dev, &ci, NULL, &r->layout)) { r->ok = 0; LOG("[RED] [vk] a pipeline layout was refused"); }
        }
        if (!r->ok) g_failed++;
    }
    return r;
}
static vks_prog *cache_find(uint64_t serial, uint32_t *slot) {
    uint32_t i = (uint32_t)(serial * 0x9E3779B97F4A7C15ull >> 50) & (PR_CAP - 1);
    for (uint32_t k = 0; k < PR_CAP; k++, i = (i + 1) & (PR_CAP - 1)) {
        if (g_pr[i] && g_pr[i]->serial == serial) { *slot = i; return g_pr[i]; }
        if (!g_pr[i]) { *slot = i; return NULL; }
    }
    *slot = PR_CAP; return NULL;
}
static void cache_put(uint32_t slot, vks_prog *r) { if (slot < PR_CAP && !g_pr[slot]) g_pr[slot] = r; }   // a full table stops caching (a program resolves again at each bind)
const vks_prog *vks_prog_get(const void *prog) {
    const prog_t *p = prog;
    if (!p || !p->serial || !VK.ok || !vks_init()) return NULL;
    pthread_mutex_lock(&g_mu);
    uint32_t slot; vks_prog *r = cache_find(p->serial, &slot);
    if (!r) { r = prog_build(p, p->serial, -1); g_progs++; cache_put(slot, r); }
    r->prog = p;
    pthread_mutex_unlock(&g_mu);
    return r->ok ? r : NULL;
}
// A program with a geometry shader, for a draw's topology (0 points, 1 lines, 2 lines with adjacency, 3 triangles, 4 triangles with adjacency)
const vks_prog *vks_prog_topology(const vks_prog *base, int cls) {
    if (!base || !base->has_gs || base->gs_class >= 0) return base;
    if (cls < 0 || cls > 4) return NULL;
    pthread_mutex_lock(&g_mu);
    uint64_t serial = base->serial ^ (GVAR * (uint64_t)(cls + 1)) ^ 0x5bd1e995ull;
    uint32_t slot; vks_prog *r = cache_find(serial, &slot);
    if (!r) {
        static int n; int say = n++ < 30;
        if (say) { LOG("[vk] building a geometry program for input topology %d (vertex %016llx, geometry %016llx, fragment %016llx)", cls, base->st[VKS_VS] ? (unsigned long long)base->st[VKS_VS]->key : 0ull,
                       base->st[VKS_GS] ? (unsigned long long)base->st[VKS_GS]->key : 0ull, base->st[VKS_FS] ? (unsigned long long)base->st[VKS_FS]->key : 0ull); fflush(stderr); }
        r = prog_build(base->prog, serial, cls); cache_put(slot, r);
        if (say) { LOG("[vk] geometry program built: %s", r->ok ? "ok" : "failed"); fflush(stderr); }
    }
    pthread_mutex_unlock(&g_mu);
    return r->ok ? r : NULL;
}
void vks_stats(uint64_t *progs, uint64_t *missing, uint64_t *failed) { *progs = g_progs; *missing = g_missing; *failed = g_failed; }
