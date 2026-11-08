// The null graphics device for Linux: the interface dm/gfx/nvn/nvn_mtl.m gives the NVN layer (hle_nvn.c), with no GPU behind it. Every submission completes at once, presents
// pace at the frame period the game asks for, and the statistics read zero. With it a game boots and runs its frame loop headless, so the CPU side of the port (rewritten code,
// HLE services, threads, memory, files) can be checked on Linux before the Vulkan backend (dm/linux/gfx_vulkan.c, to come) replaces this file.
#include "dm.h"
#include "../gfx/nvn/nvn_obj.h"
#include "front.h"
#include "gfx_vk.h"
#include <stdatomic.h>
#include <time.h>

static _Atomic uint64_t g_presents;
// A game that stops presenting (a hang, or a load that never ends) gives no crash dump: after DM_WATCHDOG_S seconds (default 20; 0 off) without a present, every guest thread's
// state is logged once (svc_thread_dump: where each one waits, or the code a spinning one runs), and again after three times as long. Pokémon Legends: Z-A stopped presenting
// as its story began, with sound still playing
static _Atomic uint64_t g_wd_last;
static uint64_t now_mono(void);
static void *present_watchdog(void *arg) {
    (void)arg; void svc_thread_dump(void);
    const char *e = getenv("DM_WATCHDOG_S"); double lim = e && *e ? atof(e) : 20; if (lim <= 0) return NULL;
    int dumps = 0, slow = 0; uint64_t seen = 0, slow_at = 0;
    const char *se = getenv("DM_WATCHDOG_SLOW"); double slow_lim = se && *se ? atof(se) : 0.4;   // a frame this long (s) is sampled: what the threads do while the game crawls (0: off)
    for (;;) {
        usleep(250000);
        uint64_t last = atomic_load(&g_wd_last), now = now_mono(), n = atomic_load(&g_presents);
        if (n != seen) { seen = n; dumps = 0; }
        double idle = (double)(now - last) / 1e9;
        // Slow frames: up to 6 samples, 15 s apart (Pokémon Legends: Z-A went from 30 fps to one frame a second in Lumiose, with the GPU idle)
        if (slow_lim > 0 && idle > slow_lim && idle < lim && slow < 6 && (!slow_at || now - slow_at > 10000000000ull)) {
            slow++; slow_at = now;
            LOG("[RED] watchdog: a slow frame (%.1f s so far since present #%llu): sample %d of the guest threads", idle, (unsigned long long)n, slow);
            svc_thread_dump();
        }
        if ((dumps == 0 && idle > lim) || (dumps == 1 && idle > 3 * lim)) {
            LOG("[RED] watchdog: no present for %.0f s (last was #%llu): the guest threads", idle, (unsigned long long)n);
            svc_thread_dump(); dumps++;
        }
    }
    return NULL;
}
static uint64_t now_mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

// Presents: the null device has no display link, so it paces itself. Each present is held until the previous one has been on "screen" for its period (the game's present interval),
// as a vsync would, so the game's frame loop runs at its own rate instead of spinning
static pthread_mutex_t g_pmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_shown_until, g_first_ns, g_last_present_ns;
static double g_times[64];
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    if (vkw_wanted()) {
        int vk_present_image(const tex_t *t, VkImage *img, uint32_t *w, uint32_t *h, int *srgb);
        VkImage img = VK_NULL_HANDLE; uint32_t w = 0, h = 0; int srgb = 0;
        if (!vk_present_image(t, &img, &w, &h, &srgb)) img = VK_NULL_HANDLE;
        else vkw_image_srgb(srgb);
        vkw_present(atomic_load(&g_presents) + 1, img, w, h, crop);
    }   // the window's FIFO swapchain paces the game (the display's vsync), as the Switch's display does
    int win = vkw_active();
    pthread_mutex_lock(&g_pmu);
    uint64_t now = now_mono();
    if (!win && g_shown_until > now) { struct timespec ts = {(time_t)(g_shown_until / 1000000000ull), (long)(g_shown_until % 1000000000ull)}; pthread_mutex_unlock(&g_pmu); clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL); pthread_mutex_lock(&g_pmu); now = now_mono(); }
    if (period_s <= 0) period_s = 1.0 / 60;
    g_shown_until = win ? 0 : now + (uint64_t)(period_s * 1e9);
    if (!g_first_ns) g_first_ns = now;
    uint64_t n = atomic_fetch_add(&g_presents, 1) + 1;
    g_times[n % 64] = (double)(now - g_first_ns) / 1e9;
    if (n == 1 || n % 600 == 0) LOG("[null gfx] present #%llu (%.1f s since the first; %.1f fps over the last %s)", (unsigned long long)n, g_times[n % 64],
                                    g_last_present_ns && n > 1 ? 600.0 / ((double)(now - g_last_present_ns) / 1e9) : 0.0, n > 1 ? "600" : "0");
    if (n % 600 == 0 || n == 1) g_last_present_ns = now;
    atomic_store(&g_wd_last, now);
    if (n == 1) { pthread_t th; if (!pthread_create(&th, NULL, present_watchdog, NULL)) pthread_detach(th); }
    pthread_mutex_unlock(&g_pmu);
    return n;
}
void mtl_present_wait(uint64_t ticket) { (void)ticket; }
double mtl_presented_time(uint64_t n) { pthread_mutex_lock(&g_pmu); double t = n && n + 64 > atomic_load(&g_presents) ? g_times[n % 64] : 0; pthread_mutex_unlock(&g_pmu); return t; }
void mtl_present_progress(uint64_t *queued, uint64_t *done) { uint64_t n = atomic_load(&g_presents); if (queued) *queued = n; if (done) *done = n; }   // presents are shown as they come (hle_vi.c frame history)
void mtl_present_layer(uint32_t w, uint32_t h) { (void)w; (void)h; }    // the window scales the picture to fit, whatever its shape
void mtl_present_screen(uint32_t w, uint32_t h) { (void)w; (void)h; }
void mtl_pool_addressed(const void *mem, uint64_t size) { (void)mem; (void)size; }   // Metal's raw global memory: the Vulkan replay copies what it reads
int mtl_addr_known(uint64_t addr, uint64_t len) { (void)addr; (void)len; return 0; }   // as mtl_addr here: no pool lookup (shaders are keyed from their own bytes)
void mtl_tex_gpu_written(const tex_t *t) { (void)t; }   // the Vulkan replay tracks GPU writes itself (gfx_vk_replay.c gpu_wrote)
void mtl_pool_stats(uint64_t s[6]) { for (int i = 0; i < 6; i++) s[i] = 0; }
int front_ctrl_support(int min, int max) { (void)min; (void)max; return -1; }   // no prompt window: the caller answers as the player closing it
int mtl_fps_target(void) {
    static int env = -2; if (env == -2) { const char *e = getenv("DM_FPS"); env = e ? atoi(e) : -1; if (env != 30 && env != 60) env = -1; }
    int nvn_game_fps(void); int g = nvn_game_fps(); return env > 0 ? env : g == 30 || g == 120 ? g : 60;
}
int mtl_display_hz(void) { return vkw_active() ? 60 : 0; }
int mtl_has_window(void) { return vkw_active(); }
int mtl_link_active(void) { return 0; }
int mtl_link_rate(void) { return 60; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)timeout_ns; struct timespec t = {0, (long)(vsyncs > 0 ? vsyncs : 1) * 16666667L}; nanosleep(&t, NULL); return 1; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)timeout_ns; struct timespec t = {0, (long)(ticks > 0 ? ticks : 1) * 16666667L}; nanosleep(&t, NULL); return 1; }
int mtl_snap_on(void) { return 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; LOG("[null gfx] screenshot requested: the null graphics device has no picture"); }

// The front end: headless (front_mac.m runs the guest on the main thread when DM_WINDOW is not set, which is all this does). The window comes with the Vulkan backend
void front_run(void (*guest)(void)) { guest(); }   // the window (DM_WINDOW=1) is made by the first present (gfx_vk_window.c)
int front_screen_max_fps(void) { return 60; }
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    (void)title; (void)sub; (void)guide; (void)max_chars;
    snprintf(out, (size_t)out_cap, "%s", initial ? initial : ""); LOG("software keyboard \"%s\": no window, the initial text is returned", title ? title : ""); return 1;
}
// Apple-only counters and the shader capture's hooks
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; }

// ---- everything else: no work and zero statistics
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) { (void)addr; (void)len; (void)buf; if (off) *off = 0; (void)off; return 0; }
void mtl_alias_drop_pool(const void *pool) { (void)pool; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { if (enc) *enc = 0; (void)enc; if (reuse) *reuse = 0; (void)reuse; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full_rw_passes) { if (load) *load = 0; (void)load; if (store) *store = 0; (void)store; if (full_rw_passes) *full_rw_passes = 0; (void)full_rw_passes; }
unsigned mtl_cpu_tex_count(void) { return 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { if (lib) *lib = 0; (void)lib; if (pso) *pso = 0; (void)pso; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; if (t0) *t0 = 0; (void)t0; if (t1) *t1 = 0; (void)t1; return 0; }
int mtl_link_stalled(void) { return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; if (target) *target = 0; (void)target; if (lead_ms) *lead_ms = 0; (void)lead_ms; if (ticks) *ticks = 0; (void)ticks; if (gap_ms) *gap_ms = 0; (void)gap_ms; return 0; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { if (uploads) *uploads = 0; (void)uploads; if (reuploads) *reuploads = 0; (void)reuploads; }
void mtl_pool_add(const void *mem, uint64_t size) { (void)mem; (void)size; }
void mtl_pool_del(const void *mem) { (void)mem; }
void mtl_pool_gpu_written(const void *pool) { (void)pool; }
void mtl_prepare(void) {  }
void mtl_prog_forget(const prog_t *p) { (void)p; }
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
void mtl_prog_release(prog_t *p) { (void)p; }
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) { (void)t; (void)level; (void)layer0; (void)r; (void)dst; (void)row; (void)img; }
void mtl_set_frame(uint64_t f) { (void)f; }
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
void mtl_stats(uint64_t *passes, uint64_t *draws) { if (passes) *passes = 0; (void)passes; if (draws) *draws = 0; (void)draws; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { if (call) *call = 0; (void)call; if (skip) *skip = 0; (void)skip; }
void mtl_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip, uint64_t *copies, uint64_t *copy_bytes) { if (resolve) *resolve = 0; (void)resolve; if (reuse) *reuse = 0; (void)reuse; if (set) *set = 0; (void)set; if (skip) *skip = 0; (void)skip; if (copies) *copies = 0; (void)copies; if (copy_bytes) *copy_bytes = 0; (void)copy_bytes; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_restop(void) {  }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
void mtl_alloc_stats(uint64_t *a, uint64_t *b, int *c) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { if (skip) *skip = 0; (void)skip; if (full) *full = 0; (void)full; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { if (lib_ns) *lib_ns = 0; (void)lib_ns; if (pso_ns) *pso_ns = 0; (void)pso_ns; }
void mtl_cshoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; }
uint64_t mtl_cut_count(void) { return 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_fb_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; if (d) *d = 0; (void)d; }
void mtl_geohoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; }
void mtl_geomesh_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; if (d) *d = 0; (void)d; if (e) *e = 0; (void)e; }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { if (draws) *draws = 0; (void)draws; if (made) *made = 0; (void)made; if (none) *none = 0; (void)none; }
void mtl_lc_stats(uint64_t *a, uint64_t *b) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; }
void mtl_mem_stat(unsigned long long a, unsigned long long b) { (void)a; (void)b; }
uint32_t mtl_present_qos(void) { return 0; }
int mtl_probe_ramp_on(void) { return 0; }
void mtl_pso_line(char *out, size_t n) { if (out && n) out[0] = 0; (void)out; (void)n; }
int mtl_pso_prebuild_tool(void) { return 0; }
void mtl_res_stats(char *out, size_t n) { if (out && n) out[0] = 0; (void)out; (void)n; }
void mtl_sm_stats(uint64_t *a, uint64_t *b, uint64_t *c) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; }
void mtl_ubd_stats(uint64_t *a, uint64_t *b) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; }
void mtl_ubp_stats(uint64_t *a, uint64_t *b, uint64_t *c) { if (a) *a = 0; (void)a; if (b) *b = 0; (void)b; if (c) *c = 0; (void)c; }
void mtl_ubp_why(uint64_t w[5]) { if (w) memset(w, 0, sizeof(*w) * 5); (void)w; }
uint64_t mtl_upload_bytes(void) { return 0; }
