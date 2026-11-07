// Linux implementations of the Apple APIs declared in dm/linux/include/dm_linux.h (and its dispatch / CoreFoundation headers), plus the small blocks runtime
// clang's -fblocks code calls. See dm_linux.h for what each stands in for.
#define _GNU_SOURCE 1
#include "dm_linux.h"
#include "dispatch/dispatch.h"
#include "CoreFoundation/CoreFoundation.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#undef fcntl

uintptr_t vm_page_size = 4096;
const struct mach_header_64 _mh_execute_header = {0xfeedfacf};
__attribute__((constructor)) static void page_size_init(void) { long p = sysconf(_SC_PAGESIZE); if (p > 0) vm_page_size = (uintptr_t)p; }

// ---------------------------------------------------------------- time
uint64_t mach_absolute_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
int mach_wait_until(uint64_t deadline) {
    struct timespec t = {(time_t)(deadline / 1000000000ull), (long)(deadline % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL) == EINTR) {}
    return 0;
}

// ---------------------------------------------------------------- memory
static int prot_of(vm_prot_t p) { return (p & VM_PROT_READ ? PROT_READ : 0) | (p & VM_PROT_WRITE ? PROT_WRITE : 0) | (p & VM_PROT_EXECUTE ? PROT_EXEC : 0); }

kern_return_t vm_allocate(mach_port_t t, vm_address_t *addr, vm_size_t size, int flags) {
    (void)t;
    // Shared anonymous pages (shmem): vm_remap(copy = FALSE) can map them a second time with mremap(old_size = 0), as Mach maps the same pages twice
    int fl = MAP_SHARED | MAP_ANONYMOUS | MAP_NORESERVE;
    if (!(flags & VM_FLAGS_ANYWHERE)) fl |= (flags & VM_FLAGS_OVERWRITE) ? MAP_FIXED : MAP_FIXED_NOREPLACE;
    void *p = mmap((flags & VM_FLAGS_ANYWHERE) ? NULL : (void *)*addr, size, PROT_READ | PROT_WRITE, fl, -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    *addr = (vm_address_t)p; return KERN_SUCCESS;
}
kern_return_t vm_deallocate(mach_port_t t, vm_address_t addr, vm_size_t size) { (void)t; return size && munmap((void *)addr, size) ? KERN_INVALID_ADDRESS : KERN_SUCCESS; }
kern_return_t vm_protect(mach_port_t t, vm_address_t addr, vm_size_t size, boolean_t set_max, vm_prot_t prot) {
    (void)t; if (set_max) return KERN_SUCCESS;   // Linux has no maximum protection
    return mprotect((void *)addr, size, prot_of(prot)) ? KERN_PROTECTION_FAILURE : KERN_SUCCESS;
}
// What protection a range has now, from /proc/self/maps (vm_remap reports it back as cur / max; mach_vm_region answers it)
static int region_of(uintptr_t a, uintptr_t *lo, uintptr_t *hi, int *prot, int *shared) {
    FILE *f = fopen("/proc/self/maps", "r"); if (!f) return 0;
    char line[512]; int found = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long l, h; char perm[8];
        if (sscanf(line, "%lx-%lx %7s", &l, &h, perm) != 3) continue;
        if (a < l) { if (lo) { *lo = l; *hi = h; *prot = (perm[0] == 'r') | (perm[1] == 'w') << 1 | (perm[2] == 'x') << 2; *shared = perm[3] == 's'; found = 2; } break; }
        if (a < h) { *lo = l; *hi = h; *prot = (perm[0] == 'r') | (perm[1] == 'w') << 1 | (perm[2] == 'x') << 2; *shared = perm[3] == 's'; found = 1; break; }
    }
    fclose(f); return found;
}
kern_return_t vm_remap(mach_port_t t, vm_address_t *to, vm_size_t size, vm_offset_t mask, int flags, mach_port_t src_task, vm_address_t from, boolean_t copy,
                       vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit) {
    (void)t; (void)mask; (void)src_task; (void)inherit;
    if (!size) return KERN_INVALID_ARGUMENT;
    uintptr_t lo, hi; int prot = PROT_READ | PROT_WRITE, shared = 0;
    if (region_of(from, &lo, &hi, &prot, &shared) != 1) return KERN_INVALID_ADDRESS;
    void *want = (flags & VM_FLAGS_ANYWHERE) ? NULL : (void *)*to, *got = MAP_FAILED;
    if (!copy && shared) {
        // Same pages a second time. mremap with old_size 0 duplicates a shared mapping (the source stays mapped)
        got = mremap((void *)from, 0, size, MREMAP_MAYMOVE | (want ? MREMAP_FIXED : 0), want);
    } else if (copy || !shared) {
        if (!copy) {
            // Private pages cannot be aliased on Linux. The callers that ask for an alias (virtual pools, cheats' writable view) map guest memory, which the Linux build keeps shared
            static _Atomic int warned; if (!atomic_exchange(&warned, 1)) fprintf(stderr, "[RED] [linux] vm_remap: an alias of private memory at %#lx was asked for: copied instead (writes will not be shared)\n", (unsigned long)from);
        }
        got = mmap(want, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (want ? MAP_FIXED : 0), -1, 0);
        if (got != MAP_FAILED) { memcpy(got, (void *)from, size); __builtin___clear_cache((char *)got, (char *)got + size); if (mprotect(got, size, prot)) {} }
    }
    if (got == MAP_FAILED) return KERN_NO_SPACE;
    *to = (vm_address_t)got;
    int c = (prot & PROT_READ ? VM_PROT_READ : 0) | (prot & PROT_WRITE ? VM_PROT_WRITE : 0) | (prot & PROT_EXEC ? VM_PROT_EXECUTE : 0);
    if (cur) *cur = c;
    if (max) *max = VM_PROT_ALL;
    return KERN_SUCCESS;
}
kern_return_t vm_read_overwrite(mach_port_t t, vm_address_t from, vm_size_t size, vm_address_t to, vm_size_t *got) {
    (void)t;
    struct iovec l = {(void *)to, size}, r = {(void *)from, size};
    ssize_t n = process_vm_readv(getpid(), &l, 1, &r, 1, 0);   // fails (EFAULT) instead of faulting on an unmapped address
    if (n < 0) { if (got) *got = 0; return KERN_INVALID_ADDRESS; }
    if (got) *got = (vm_size_t)n;
    return (vm_size_t)n == size ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
kern_return_t vm_write(mach_port_t t, vm_address_t to, vm_offset_t from, mach_msg_type_number_t size) {
    (void)t;
    struct iovec l = {(void *)from, size}, r = {(void *)to, size};
    return process_vm_writev(getpid(), &l, 1, &r, 1, 0) == (ssize_t)size ? KERN_SUCCESS : KERN_PROTECTION_FAILURE;
}
kern_return_t mach_vm_region(mach_port_t t, mach_vm_address_t *addr, mach_vm_size_t *size, int flavor, vm_region_info_t info, mach_msg_type_number_t *cnt, mach_port_t *obj) {
    (void)t; (void)cnt; if (obj) *obj = 0;
    if (flavor != VM_REGION_BASIC_INFO_64) return KERN_INVALID_ARGUMENT;
    uintptr_t lo, hi; int prot, shared;
    if (!region_of(*addr, &lo, &hi, &prot, &shared)) return KERN_INVALID_ADDRESS;   // like Mach, the next region at or above addr
    vm_region_basic_info_data_64_t *bi = (vm_region_basic_info_data_64_t *)info; memset(bi, 0, sizeof *bi);
    bi->protection = prot; bi->max_protection = VM_PROT_ALL; bi->shared = shared;
    *addr = lo; *size = hi - lo; return KERN_SUCCESS;
}

// ---------------------------------------------------------------- task and thread information
static uint64_t status_kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r"); if (!f) return 0;
    char line[256]; uint64_t v = 0; size_t k = strlen(key);
    while (fgets(line, sizeof line, f)) if (!strncmp(line, key, k) && line[k] == ':') { v = strtoull(line + k + 1, NULL, 10); break; }
    fclose(f); return v;
}
kern_return_t task_info(mach_port_t t, int flavor, task_info_t info, mach_msg_type_number_t *cnt) {
    (void)t; (void)cnt;
    if (flavor == TASK_VM_INFO) {
        task_vm_info_data_t *vi = (task_vm_info_data_t *)info; memset(vi, 0, sizeof *vi);
        vi->phys_footprint = status_kb("VmRSS") << 10; vi->resident_size = vi->phys_footprint; vi->resident_size_peak = status_kb("VmHWM") << 10;
        vi->internal = (status_kb("RssAnon") + status_kb("RssShmem")) << 10; vi->external = status_kb("RssFile") << 10; vi->virtual_size = status_kb("VmSize") << 10;
        vi->page_size = vm_page_size;
        return KERN_SUCCESS;
    }
    return KERN_FAILURE;   // power and energy counters: no Linux counterpart; the callers skip their lines
}
kern_return_t task_threads(mach_port_t t, thread_act_array_t *list, mach_msg_type_number_t *n) { (void)t; *list = NULL; *n = 0; return KERN_FAILURE; }
mach_port_t mach_thread_self(void) { return (mach_port_t)syscall(SYS_gettid); }
mach_port_t pthread_mach_thread_np(pthread_t th) { return pthread_equal(th, pthread_self()) ? mach_thread_self() : 0; }
kern_return_t thread_info(mach_port_t th, int flavor, thread_info_t info, mach_msg_type_number_t *cnt) {
    (void)cnt;
    if (flavor != THREAD_BASIC_INFO || !th) return KERN_FAILURE;
    char p[64]; snprintf(p, sizeof p, "/proc/self/task/%u/stat", th);
    FILE *f = fopen(p, "r"); if (!f) return KERN_FAILURE;
    char b[1024]; size_t n = fread(b, 1, sizeof b - 1, f); fclose(f); b[n] = 0;
    char *e = strrchr(b, ')'); if (!e) return KERN_FAILURE;
    char st = 0; unsigned long ut = 0, stt = 0;
    if (sscanf(e + 2, "%c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &st, &ut, &stt) != 3) return KERN_FAILURE;
    long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    thread_basic_info_data_t *bi = (thread_basic_info_data_t *)info; memset(bi, 0, sizeof *bi);
    bi->user_time.seconds = (int)(ut / hz); bi->user_time.microseconds = (int)(ut % hz * 1000000 / hz);
    bi->system_time.seconds = (int)(stt / hz); bi->system_time.microseconds = (int)(stt % hz * 1000000 / hz);
    bi->run_state = st == 'R' ? TH_STATE_RUNNING : st == 'D' ? TH_STATE_UNINTERRUPTIBLE : st == 'T' || st == 't' ? TH_STATE_STOPPED : TH_STATE_WAITING;
    return KERN_SUCCESS;
}
// Suspending another thread to read its registers (hang dumps: what a thread spinning in guest code is running). A real-time signal stops it in a handler that copies its registers
// and waits there until thread_resume. One thread at a time (the dumps walk the threads one by one). A thread in a system call is taken out of it (EINTR / restart): only the
// watchdog's dumps use this, and the process exits after them
#include <signal.h>
#include <ucontext.h>
#define DM_SUSPEND_SIG (SIGRTMIN + 3)
static _Atomic int g_sus_tid, g_sus_got, g_sus_resume, g_sus_left;
static struct __darwin_arm_thread_state64 g_sus_st;
static void sus_handler(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si;
    if (atomic_load(&g_sus_tid) != (int)syscall(SYS_gettid)) return;
    ucontext_t *uc = uc_;
    memcpy(&g_sus_st, &uc->uc_mcontext.regs[0], sizeof g_sus_st.__x + 3 * 8);   // x0-x28, fp, lr
    g_sus_st.__sp = uc->uc_mcontext.sp; g_sus_st.__pc = uc->uc_mcontext.pc; g_sus_st.__cpsr = (uint32_t)uc->uc_mcontext.pstate;
    atomic_store(&g_sus_got, 1);
    for (int i = 0; i < 2000 && !atomic_load(&g_sus_resume); i++) { struct timespec t = {0, 1000000}; nanosleep(&t, NULL); }   // 2 s at most
    atomic_store(&g_sus_left, 1);
}
kern_return_t thread_suspend(mach_port_t th) {
    static int installed;
    if (!installed) { struct sigaction sa = {.sa_sigaction = sus_handler, .sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK}; sigemptyset(&sa.sa_mask); sigaction(DM_SUSPEND_SIG, &sa, NULL); installed = 1; }
    if (!th || (int)th == (int)syscall(SYS_gettid)) return KERN_FAILURE;
    atomic_store(&g_sus_got, 0); atomic_store(&g_sus_resume, 0); atomic_store(&g_sus_left, 0); atomic_store(&g_sus_tid, (int)th);
    if (syscall(SYS_tgkill, getpid(), (int)th, DM_SUSPEND_SIG)) { atomic_store(&g_sus_tid, 0); return KERN_FAILURE; }
    for (int i = 0; i < 200 && !atomic_load(&g_sus_got); i++) { struct timespec t = {0, 1000000}; nanosleep(&t, NULL); }
    if (!atomic_load(&g_sus_got)) { atomic_store(&g_sus_resume, 1); atomic_store(&g_sus_tid, 0); return KERN_FAILURE; }   // blocked the signal, or did not run in 200 ms
    return KERN_SUCCESS;
}
kern_return_t thread_resume(mach_port_t th) {
    (void)th; atomic_store(&g_sus_resume, 1);
    for (int i = 0; i < 200 && !atomic_load(&g_sus_left); i++) { struct timespec t = {0, 1000000}; nanosleep(&t, NULL); }
    atomic_store(&g_sus_tid, 0); return KERN_SUCCESS;
}
kern_return_t thread_get_state(mach_port_t th, int flavor, thread_state_t st, mach_msg_type_number_t *cnt) {
    (void)cnt; if (flavor != ARM_THREAD_STATE64 || (int)th != atomic_load(&g_sus_tid) || !atomic_load(&g_sus_got)) return KERN_FAILURE;
    memcpy(st, &g_sus_st, sizeof g_sus_st); return KERN_SUCCESS;
}
kern_return_t thread_policy_set(mach_port_t th, int flavor, thread_policy_t p, mach_msg_type_number_t cnt) { (void)th; (void)flavor; (void)p; (void)cnt; return KERN_NOT_SUPPORTED; }
kern_return_t thread_policy_get(mach_port_t th, int flavor, thread_policy_t p, mach_msg_type_number_t *cnt, boolean_t *def) { (void)th; (void)flavor; (void)p; (void)cnt; if (def) *def = 1; return KERN_NOT_SUPPORTED; }

int proc_pid_rusage(int pid, int flavor, rusage_info_t *buffer) {
    (void)pid; if (flavor != RUSAGE_INFO_V4) return -1;
    struct rusage_info_v4 *r = (struct rusage_info_v4 *)buffer; memset(r, 0, sizeof *r);
    FILE *f = fopen("/proc/self/io", "r");
    if (f) { char line[128]; unsigned long long v;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "read_bytes: %llu", &v) == 1) r->ri_diskio_bytesread = v;
            else if (sscanf(line, "write_bytes: %llu", &v) == 1) r->ri_diskio_byteswritten = v;
            else if (sscanf(line, "wchar: %llu", &v) == 1) r->ri_logical_writes = v;
        }
        fclose(f); }
    r->ri_lifetime_max_phys_footprint = status_kb("VmHWM") << 10; r->ri_phys_footprint = status_kb("VmRSS") << 10;
    return 0;
}
int sysctlbyname(const char *name, void *old, size_t *oldlen, void *new_, size_t newlen) {
    (void)new_; (void)newlen;
    if (!strcmp(name, "hw.model") || !strcmp(name, "hw.machine")) {
        char m[128] = "linux-aarch64"; FILE *f = fopen("/sys/firmware/devicetree/base/model", "r");
        if (!f) f = fopen("/sys/class/dmi/id/product_name", "r");
        if (f) { size_t n = fread(m, 1, sizeof m - 1, f); m[n] = 0; fclose(f); for (char *c = m; *c; c++) if (*c == '\n') *c = 0; }
        size_t n = strlen(m) + 1; if (!old || *oldlen < n) { *oldlen = n; return old ? -1 : 0; }
        memcpy(old, m, n); *oldlen = n; return 0;
    }
    if (!strcmp(name, "hw.ncpu") || !strcmp(name, "hw.logicalcpu") || !strcmp(name, "hw.physicalcpu")) {
        if (!old || *oldlen < sizeof(int)) return -1;
        *(int *)old = (int)sysconf(_SC_NPROCESSORS_ONLN); *oldlen = sizeof(int); return 0;
    }
    if (!strcmp(name, "hw.memsize")) {
        if (!old || *oldlen < sizeof(uint64_t)) return -1;
        *(uint64_t *)old = (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE); *oldlen = sizeof(uint64_t); return 0;
    }
    errno = ENOENT; return -1;   // hw.nperflevels and the rest: the callers take their defaults
}

// ---------------------------------------------------------------- pthread extras
static _Thread_local qos_class_t g_qos = QOS_CLASS_DEFAULT;
int pthread_set_qos_class_self_np(qos_class_t q, int rel) { (void)rel; g_qos = q; return 0; }   // Linux has no QoS classes; the scheduler treats every thread the same
int pthread_get_qos_class_np(pthread_t t, qos_class_t *q, int *rel) { (void)t; if (q) *q = g_qos; if (rel) *rel = 0; return 0; }
int pthread_attr_set_qos_class_np(pthread_attr_t *a, qos_class_t q, int rel) { (void)a; (void)q; (void)rel; return 0; }
qos_class_t qos_class_self(void) { return g_qos; }
void *pthread_get_stackaddr_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t n = 0;
    if (pthread_getattr_np(t, &a)) return NULL;
    pthread_attr_getstack(&a, &lo, &n); pthread_attr_destroy(&a);
    return (char *)lo + n;   // Apple returns the top (highest address)
}
size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a; void *lo = NULL; size_t n = 0;
    if (pthread_getattr_np(t, &a)) return 0;
    pthread_attr_getstack(&a, &lo, &n); pthread_attr_destroy(&a);
    return n;
}
int pthread_cond_timedwait_relative_np(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel) {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);   // a default condition variable waits on CLOCK_REALTIME
    t.tv_sec += rel->tv_sec; t.tv_nsec += rel->tv_nsec;
    if (t.tv_nsec >= 1000000000L) { t.tv_sec++; t.tv_nsec -= 1000000000L; }
    return pthread_cond_timedwait(c, m, &t);
}
int pthread_threadid_np(pthread_t t, uint64_t *id) {
    if (!id) return EINVAL;
    *id = (!t || pthread_equal(t, pthread_self())) ? (uint64_t)syscall(SYS_gettid) : (uint64_t)t;
    return 0;
}
#undef pthread_setname_np
int dm_pthread_setname_self(const char *name) {
    char n[16]; snprintf(n, sizeof n, "%s", name);   // Linux allows 15 characters
    return pthread_setname_np(pthread_self(), n);
}

// ---------------------------------------------------------------- libc differences
static void msort(char *a, char *tmp, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
    if (n < 2) return;
    size_t h = n / 2;
    msort(a, tmp, h, sz, cmp); msort(a + h * sz, tmp, n - h, sz, cmp);
    size_t i = 0, j = h, k = 0;
    while (i < h && j < n) { if (cmp(a + j * sz, a + i * sz) < 0) memcpy(tmp + k++ * sz, a + j++ * sz, sz); else memcpy(tmp + k++ * sz, a + i++ * sz, sz); }
    while (i < h) memcpy(tmp + k++ * sz, a + i++ * sz, sz);
    while (j < n) memcpy(tmp + k++ * sz, a + j++ * sz, sz);
    memcpy(a, tmp, n * sz);
}
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *)) {
    if (n < 2) return 0;
    char *tmp = malloc(n * size); if (!tmp) { errno = ENOMEM; return -1; }
    msort(base, tmp, n, size, cmp); free(tmp); return 0;
}
int dm_fcntl(int fd, int cmd, ...) {
    va_list ap; va_start(ap, cmd); void *arg = va_arg(ap, void *); va_end(ap);
    if (cmd == F_GETPATH) {
        char p[64]; snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(p, (char *)arg, 1023); if (n < 0) return -1;
        ((char *)arg)[n] = 0; return 0;
    }
    if (cmd == F_NOCACHE) { if ((intptr_t)arg) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); return 0; }
    if (cmd == F_FULLFSYNC) return fsync(fd);
    return fcntl(fd, cmd, arg);
}

// ---------------------------------------------------------------- Mach-O sections → ELF sections
// The registration tables. Each is a section of its own name: the linker defines __start_<name> and __stop_<name> for it (weak: a table no file uses is empty)
extern const char __start_swhle[] __attribute__((weak)), __stop_swhle[] __attribute__((weak));
extern const char __start_swforce[] __attribute__((weak)), __stop_swforce[] __attribute__((weak));
extern const char __start_swnvn[] __attribute__((weak)), __stop_swnvn[] __attribute__((weak));
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *seg, const char *sect, unsigned long *size) {
    (void)mh; (void)seg;
    const char *b = NULL, *e = NULL;
    if (!strcmp(sect, "swhle")) b = __start_swhle, e = __stop_swhle;
    else if (!strcmp(sect, "swforce")) b = __start_swforce, e = __stop_swforce;
    else if (!strcmp(sect, "swnvn")) b = __start_swnvn, e = __stop_swnvn;
    if (!b || !e) { if (size) *size = 0; return NULL; }   // embedded game modules: the Linux build loads them from files (loader.c's non-iPhone branch)
    if (size) *size = (unsigned long)(e - b);
    return (uint8_t *)b;
}

// ---------------------------------------------------------------- SHA-256 (CommonCrypto)
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, k, t1, t2;
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) w[i] = w[i - 16] + (ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] + (ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10));
    a = h[0]; b = h[1]; c = h[2]; d = h[3]; e = h[4]; f = h[5]; g = h[6]; k = h[7];
    for (int i = 0; i < 64; i++) {
        t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}
int CC_SHA256_Init(CC_SHA256_CTX *c) {
    static const uint32_t iv[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h, iv, sizeof iv); c->len = 0; c->n = 0; return 1;
}
int CC_SHA256_Update(CC_SHA256_CTX *c, const void *data, CC_LONG len) {
    const uint8_t *p = data; c->len += len;
    while (len) {
        if (!c->n && len >= 64) { sha_block(c->h, p); p += 64; len -= 64; continue; }
        uint32_t k = 64 - c->n < len ? 64 - c->n : len;
        memcpy(c->buf + c->n, p, k); c->n += k; p += k; len -= k;
        if (c->n == 64) { sha_block(c->h, c->buf); c->n = 0; }
    }
    return 1;
}
int CC_SHA256_Final(unsigned char *md, CC_SHA256_CTX *c) {
    uint64_t bits = c->len * 8; uint8_t pad = 0x80, z = 0;
    CC_SHA256_Update(c, &pad, 1);
    while (c->n != 56) CC_SHA256_Update(c, &z, 1);
    uint8_t l[8]; for (int i = 0; i < 8; i++) l[i] = (uint8_t)(bits >> (56 - 8 * i));
    CC_SHA256_Update(c, l, 8);
    for (int i = 0; i < 8; i++) { md[4 * i] = (uint8_t)(c->h[i] >> 24); md[4 * i + 1] = (uint8_t)(c->h[i] >> 16); md[4 * i + 2] = (uint8_t)(c->h[i] >> 8); md[4 * i + 3] = (uint8_t)c->h[i]; }
    return 1;
}
unsigned char *CC_SHA256(const void *data, CC_LONG len, unsigned char *md) { CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, data, len); CC_SHA256_Final(md, &c); return md; }

// ---------------------------------------------------------------- CoreFoundation: a pointer-keyed dictionary
struct dm_cf_dict { const void **k, **v; size_t n, cap; pthread_mutex_t mu; };
static size_t ph(const void *p, size_t cap) { uint64_t x = (uint64_t)(uintptr_t)p; x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; return (size_t)x & (cap - 1); }
#define TOMB ((const void *)~(uintptr_t)0)
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *kc, const void *vc) {
    (void)a; (void)cap; (void)kc; (void)vc;
    struct dm_cf_dict *d = calloc(1, sizeof *d); d->cap = 64; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v); pthread_mutex_init(&d->mu, NULL);
    return d;
}
static long find(const struct dm_cf_dict *d, const void *key) {
    for (size_t i = ph(key, d->cap), s = 0; s < d->cap; i = (i + 1) & (d->cap - 1), s++) { if (!d->k[i]) return -1; if (d->k[i] == key) return (long)i; }
    return -1;
}
static void put(struct dm_cf_dict *d, const void *key, const void *val) {
    size_t i = ph(key, d->cap); while (d->k[i] && d->k[i] != TOMB && d->k[i] != key) i = (i + 1) & (d->cap - 1);
    if (d->k[i] != key) d->n++;
    d->k[i] = key; d->v[i] = val;
}
const void *CFDictionaryGetValue(CFDictionaryRef dc, const void *key) {
    struct dm_cf_dict *d = (struct dm_cf_dict *)dc; pthread_mutex_lock(&d->mu);
    long i = find(d, key); const void *v = i < 0 ? NULL : d->v[i];
    pthread_mutex_unlock(&d->mu); return v;
}
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *val) {
    pthread_mutex_lock(&d->mu);
    long i = find(d, key);
    if (i >= 0) d->v[i] = val;
    else {
        if ((d->n + 1) * 2 > d->cap) {
            const void **ok = d->k, **ov = d->v; size_t oc = d->cap;
            d->cap *= 2; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v); d->n = 0;
            for (size_t j = 0; j < oc; j++) if (ok[j] && ok[j] != TOMB) put(d, ok[j], ov[j]);
            free(ok); free(ov);
        }
        put(d, key, val);
    }
    pthread_mutex_unlock(&d->mu);
}
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key) {
    pthread_mutex_lock(&d->mu);
    long i = find(d, key); if (i >= 0) { d->k[i] = TOMB; d->v[i] = NULL; d->n--; }
    pthread_mutex_unlock(&d->mu);
}
CFIndex CFDictionaryGetCount(CFDictionaryRef d) { return (CFIndex)d->n; }
void CFRelease(CFTypeRef o) { (void)o; }   // the dictionaries live as long as the host
CFTypeRef CFRetain(CFTypeRef o) { return o; }

// ---------------------------------------------------------------- blocks runtime (the part clang's -fblocks code calls)
// A block literal: isa, flags, reserved, invoke, descriptor {reserved, size, copy helper, dispose helper}. Stack blocks are copied to the heap with their captures.
struct Block_descriptor { unsigned long reserved, size; void (*copy)(void *dst, const void *src); void (*dispose)(const void *); };
struct Block_layout { void *isa; int flags, reserved; void (*invoke)(void *, ...); struct Block_descriptor *descriptor; };
enum { BLOCK_REFCOUNT_MASK = 0xfffe, BLOCK_NEEDS_FREE = 1 << 24, BLOCK_HAS_COPY_DISPOSE = 1 << 25, BLOCK_IS_GLOBAL = 1 << 28 };
enum { BLOCK_FIELD_IS_OBJECT = 3, BLOCK_FIELD_IS_BLOCK = 7, BLOCK_FIELD_IS_BYREF = 8, BLOCK_FIELD_IS_WEAK = 16, BLOCK_BYREF_CALLER = 128 };
void *_NSConcreteStackBlock[32], *_NSConcreteGlobalBlock[32], *_NSConcreteMallocBlock[32];
struct Block_byref { void *isa; struct Block_byref *forwarding; int flags, size; void (*keep)(void *, void *); void (*destroy)(void *); };
void *_Block_copy(const void *arg) {
    struct Block_layout *b = (struct Block_layout *)arg;
    if (!b) return NULL;
    if (b->flags & BLOCK_IS_GLOBAL) return b;
    if (b->flags & BLOCK_NEEDS_FREE) { __atomic_add_fetch(&b->flags, 2, __ATOMIC_RELAXED); return b; }
    struct Block_layout *h = malloc(b->descriptor->size); if (!h) return NULL;
    memcpy(h, b, b->descriptor->size);
    h->isa = _NSConcreteMallocBlock; h->flags = (h->flags & ~BLOCK_REFCOUNT_MASK) | BLOCK_NEEDS_FREE | 2;
    if (b->flags & BLOCK_HAS_COPY_DISPOSE) b->descriptor->copy(h, b);
    return h;
}
void _Block_release(const void *arg) {
    struct Block_layout *b = (struct Block_layout *)arg;
    if (!b || !(b->flags & BLOCK_NEEDS_FREE)) return;
    if ((__atomic_sub_fetch(&b->flags, 2, __ATOMIC_ACQ_REL) & BLOCK_REFCOUNT_MASK) == 0) {
        if (b->flags & BLOCK_HAS_COPY_DISPOSE) b->descriptor->dispose(b);
        free(b);
    }
}
void _Block_object_assign(void *dst, const void *obj, const int flags) {
    switch (flags & (BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_BLOCK | BLOCK_FIELD_IS_OBJECT)) {
    case BLOCK_FIELD_IS_BLOCK: *(void **)dst = _Block_copy(obj); break;
    case BLOCK_FIELD_IS_BYREF: {   // a __block variable: moved to the heap once, shared by every copy
        struct Block_byref *src = ((struct Block_byref *)obj)->forwarding;
        if (src->isa == NULL && !(src->flags & BLOCK_NEEDS_FREE)) {
            struct Block_byref *c = malloc((size_t)src->size); memcpy(c, src, (size_t)src->size);
            c->forwarding = c; src->forwarding = c; c->flags |= BLOCK_NEEDS_FREE | 4;
            if (src->flags & BLOCK_HAS_COPY_DISPOSE) src->keep(c, src);
            *(void **)dst = c;
        } else { if (src->flags & BLOCK_NEEDS_FREE) __atomic_add_fetch(&src->flags, 2, __ATOMIC_RELAXED); *(void **)dst = src; }
        break; }
    default: *(const void **)dst = obj; break;   // objects: no Objective-C on Linux, kept as plain pointers
    }
}
void _Block_object_dispose(const void *obj, const int flags) {
    if ((flags & (BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_BLOCK)) == BLOCK_FIELD_IS_BLOCK) _Block_release(obj);
    else if ((flags & BLOCK_FIELD_IS_BYREF) == BLOCK_FIELD_IS_BYREF) {
        struct Block_byref *b = ((struct Block_byref *)obj)->forwarding;
        if ((b->flags & BLOCK_NEEDS_FREE) && (__atomic_sub_fetch(&b->flags, 2, __ATOMIC_ACQ_REL) & BLOCK_REFCOUNT_MASK) == 2) {
            if (b->flags & BLOCK_HAS_COPY_DISPOSE) b->destroy(b);
            free(b);
        }
    }
}
static void block_run(const void *b) { ((struct Block_layout *)b)->invoke((void *)b); }

// ---------------------------------------------------------------- dispatch
typedef struct job { struct job *next; void *ctx; dispatch_function_t f; const void *blk; uint64_t when; } job;
struct dm_dispatch_queue { pthread_mutex_t mu; pthread_cond_t cv; job *head, *tail; pthread_t th; int started; char label[64]; };
static void *queue_main(void *arg) {
    dispatch_queue_t q = arg; dm_pthread_setname_self(q->label);
    for (;;) {
        pthread_mutex_lock(&q->mu);
        while (!q->head) pthread_cond_wait(&q->cv, &q->mu);
        job *j = q->head; q->head = j->next; if (!q->head) q->tail = NULL;
        pthread_mutex_unlock(&q->mu);
        if (j->when) mach_wait_until(j->when);
        if (j->blk) { block_run(j->blk); _Block_release(j->blk); } else j->f(j->ctx);
        free(j);
    }
    return NULL;
}
dispatch_queue_t dispatch_queue_create(const char *label, dispatch_queue_attr_t attr) {
    (void)attr;   // every queue is serial, which also keeps a concurrent queue's work correct (if slower)
    dispatch_queue_t q = calloc(1, sizeof *q); pthread_mutex_init(&q->mu, NULL); pthread_cond_init(&q->cv, NULL);
    snprintf(q->label, sizeof q->label, "%s", label ? label : "dispatch");
    return q;
}
static void enqueue(dispatch_queue_t q, job *j) {
    pthread_mutex_lock(&q->mu);
    if (!q->started) { q->started = 1; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&q->th, &a, queue_main, q); pthread_attr_destroy(&a); }
    if (q->tail) q->tail->next = j; else q->head = j;
    q->tail = j; pthread_cond_signal(&q->cv); pthread_mutex_unlock(&q->mu);
}
static dispatch_queue_t g_global_queue;
static void global_queue_init(void) { g_global_queue = dispatch_queue_create("dispatch.global", 0); }
dispatch_queue_t dispatch_get_global_queue(long qos, unsigned long flags) {
    (void)qos; (void)flags; static pthread_once_t o = PTHREAD_ONCE_INIT;
    pthread_once(&o, global_queue_init);
    return g_global_queue;
}
dispatch_queue_t dispatch_get_main_queue(void) { return dispatch_get_global_queue(0, 0); }
void dispatch_async(dispatch_queue_t q, dispatch_block_t b) { job *j = calloc(1, sizeof *j); j->blk = _Block_copy((const void *)b); enqueue(q, j); }
void dispatch_async_f(dispatch_queue_t q, void *ctx, dispatch_function_t f) { job *j = calloc(1, sizeof *j); j->ctx = ctx; j->f = f; enqueue(q, j); }
void dispatch_after(dispatch_time_t when, dispatch_queue_t q, dispatch_block_t b) {
    // On the queue's own thread the wait holds the queue; the host only delays one-off work this way (a deferred quit), so the order this imposes is harmless
    job *j = calloc(1, sizeof *j); j->blk = _Block_copy((const void *)b); j->when = when ? when : 1; enqueue(q, j);
}
struct sync_wait { pthread_mutex_t mu; pthread_cond_t cv; int done; void *ctx; dispatch_function_t f; const void *blk; };
static void sync_trampoline(void *arg) {
    struct sync_wait *w = arg;
    if (w->blk) block_run(w->blk); else w->f(w->ctx);
    pthread_mutex_lock(&w->mu); w->done = 1; pthread_cond_signal(&w->cv); pthread_mutex_unlock(&w->mu);
}
static void sync_run(dispatch_queue_t q, struct sync_wait *w) {
    if (q->started && pthread_equal(q->th, pthread_self())) { w->done = 0; sync_trampoline(w); return; }
    pthread_mutex_init(&w->mu, NULL); pthread_cond_init(&w->cv, NULL); w->done = 0;
    dispatch_async_f(q, w, sync_trampoline);
    pthread_mutex_lock(&w->mu); while (!w->done) pthread_cond_wait(&w->cv, &w->mu); pthread_mutex_unlock(&w->mu);
}
void dispatch_sync(dispatch_queue_t q, dispatch_block_t b) { struct sync_wait w = {.blk = (const void *)b}; sync_run(q, &w); }
void dispatch_sync_f(dispatch_queue_t q, void *ctx, dispatch_function_t f) { struct sync_wait w = {.ctx = ctx, .f = f}; sync_run(q, &w); }
dispatch_time_t dispatch_time(dispatch_time_t base, int64_t delta) {
    if (base == DISPATCH_TIME_FOREVER) return base;
    uint64_t t = base ? base : mach_absolute_time();
    return delta < 0 && (uint64_t)-delta > t ? 1 : t + (uint64_t)delta;
}
static pthread_mutex_t g_once_mu = PTHREAD_MUTEX_INITIALIZER;
void dispatch_once(dispatch_once_t *once, dispatch_block_t b) {
    if (__atomic_load_n(once, __ATOMIC_ACQUIRE) == ~0l) return;
    pthread_mutex_lock(&g_once_mu);
    if (*once != ~0l) { block_run((const void *)b); __atomic_store_n(once, ~0l, __ATOMIC_RELEASE); }
    pthread_mutex_unlock(&g_once_mu);
}
void dispatch_once_f(dispatch_once_t *once, void *ctx, dispatch_function_t f) {
    if (__atomic_load_n(once, __ATOMIC_ACQUIRE) == ~0l) return;
    pthread_mutex_lock(&g_once_mu);
    if (*once != ~0l) { f(ctx); __atomic_store_n(once, ~0l, __ATOMIC_RELEASE); }
    pthread_mutex_unlock(&g_once_mu);
}
struct dm_dispatch_sema { pthread_mutex_t mu; pthread_cond_t cv; long v; };
dispatch_semaphore_t dispatch_semaphore_create(long value) {
    dispatch_semaphore_t s = calloc(1, sizeof *s); pthread_mutex_init(&s->mu, NULL);
    pthread_condattr_t a; pthread_condattr_init(&a); pthread_condattr_setclock(&a, CLOCK_MONOTONIC); pthread_cond_init(&s->cv, &a); pthread_condattr_destroy(&a);
    s->v = value; return s;
}
long dispatch_semaphore_signal(dispatch_semaphore_t s) {
    pthread_mutex_lock(&s->mu); long woke = s->v++ < 0; pthread_cond_signal(&s->cv); pthread_mutex_unlock(&s->mu); return woke;
}
long dispatch_semaphore_wait(dispatch_semaphore_t s, dispatch_time_t timeout) {
    pthread_mutex_lock(&s->mu);
    while (s->v <= 0) {
        if (timeout == DISPATCH_TIME_FOREVER) pthread_cond_wait(&s->cv, &s->mu);
        else {
            struct timespec t = {(time_t)(timeout / 1000000000ull), (long)(timeout % 1000000000ull)};
            if (pthread_cond_timedwait(&s->cv, &s->mu, &t) == ETIMEDOUT && s->v <= 0) { pthread_mutex_unlock(&s->mu); return 1; }
        }
    }
    s->v--; pthread_mutex_unlock(&s->mu); return 0;
}
// Signal sources: a thread waits for the signal with sigwait and runs the handler on the source's queue
struct dm_dispatch_source_type { int kind; };
const struct dm_dispatch_source_type dm_source_type_signal = {1};
struct dm_dispatch_source { int sig; dispatch_queue_t q; const void *handler; };
dispatch_source_t dispatch_source_create(dispatch_source_type_t type, uintptr_t handle, unsigned long mask, dispatch_queue_t q) {
    (void)mask; if (type != &dm_source_type_signal) return NULL;
    dispatch_source_t s = calloc(1, sizeof *s); s->sig = (int)handle; s->q = q; return s;
}
void dispatch_source_set_event_handler(dispatch_source_t s, dispatch_block_t b) { if (s) s->handler = _Block_copy((const void *)b); }
static void *signal_wait(void *arg) {
    dispatch_source_t s = arg; sigset_t set; sigemptyset(&set); sigaddset(&set, s->sig);
    for (;;) { int got; if (sigwait(&set, &got)) continue; if (s->handler) dispatch_async(s->q, (dispatch_block_t)s->handler); }
    return NULL;
}
void dispatch_resume(void *object) {
    dispatch_source_t s = object; if (!s || !s->sig) return;
    // The signal must be blocked in every thread for sigwait to take it; the host calls this at start, before its other threads exist, so they inherit the mask
    sigset_t set; sigemptyset(&set); sigaddset(&set, s->sig); pthread_sigmask(SIG_BLOCK, &set, NULL);
    pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, signal_wait, s); pthread_attr_destroy(&a);
}
void dispatch_release(void *object) { (void)object; }
void dispatch_retain(void *object) { (void)object; }

// ---------------------------------------------------------------- CF callbacks, sysctl, files
const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
#include "sys/sysctl.h"
#include "sys/clonefile.h"
#include <linux/magic.h>
int sysctl(int *mib, unsigned n, void *old, size_t *oldlen, void *new_, size_t newlen) {
    (void)new_; (void)newlen;
    if (n != 4 || mib[0] != CTL_KERN || mib[1] != KERN_PROC || mib[2] != KERN_PROC_PID || !old || *oldlen < sizeof(struct kinfo_proc)) { errno = ENOTSUP; return -1; }
    // starttime (field 22 of /proc/self/stat) is in clock ticks since boot; the wall clock at boot is now - uptime
    FILE *f = fopen("/proc/self/stat", "r"); if (!f) return -1;
    char b[1024]; size_t k = fread(b, 1, sizeof b - 1, f); fclose(f); b[k] = 0;
    char *e = strrchr(b, ')'); unsigned long long st = 0;
    if (!e || sscanf(e + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %*u %*u %*d %*d %*d %*d %*d %*d %llu", &st) != 1) return -1;
    long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    struct timespec up, now; clock_gettime(CLOCK_BOOTTIME, &up); clock_gettime(CLOCK_REALTIME, &now);
    double boot = (double)now.tv_sec + now.tv_nsec / 1e9 - ((double)up.tv_sec + up.tv_nsec / 1e9), start = boot + (double)st / (double)hz;
    struct kinfo_proc *kp = old; memset(kp, 0, sizeof *kp);
    kp->kp_proc.p_starttime.tv_sec = (time_t)start; kp->kp_proc.p_starttime.tv_usec = (suseconds_t)((start - (double)(time_t)start) * 1e6);
    *oldlen = sizeof *kp; return 0;
}
#include <dirent.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include "sys/xattr.h"
#undef getxattr
#undef setxattr
#undef removexattr
static void user_name(char *out, size_t n, const char *name) { snprintf(out, n, strncmp(name, "user.", 5) ? "user.%s" : "%s", name); }
ssize_t dm_getxattr(const char *p, const char *name, void *v, size_t n, uint32_t pos, int opt) {
    (void)pos; char nm[256]; user_name(nm, sizeof nm, name);
    return (opt & XATTR_NOFOLLOW) ? lgetxattr(p, nm, v, n) : getxattr(p, nm, v, n);
}
int dm_setxattr(const char *p, const char *name, const void *v, size_t n, uint32_t pos, int opt) {
    (void)pos; char nm[256]; user_name(nm, sizeof nm, name);
    return (opt & XATTR_NOFOLLOW) ? lsetxattr(p, nm, v, n, 0) : setxattr(p, nm, v, n, 0);
}
int dm_removexattr(const char *p, const char *name, int opt) {
    char nm[256]; user_name(nm, sizeof nm, name);
    return (opt & XATTR_NOFOLLOW) ? lremovexattr(p, nm) : removexattr(p, nm);
}
static int copy_xattrs(const char *src, const char *dst) {
    char list[4096]; ssize_t n = llistxattr(src, list, sizeof list);
    for (ssize_t i = 0; i < n; i += (ssize_t)strlen(list + i) + 1) {
        if (strncmp(list + i, "user.", 5)) continue;
        char v[4096]; ssize_t k = lgetxattr(src, list + i, v, sizeof v);
        if (k >= 0 && lsetxattr(dst, list + i, v, (size_t)k, 0)) return -1;
    }
    return 0;
}
static int clone_one(const char *src, const char *dst) {
    struct stat st; if (lstat(src, &st)) return -1;
    if (S_ISDIR(st.st_mode)) {
        if (mkdir(dst, st.st_mode & 07777)) return -1;
        DIR *d = opendir(src); if (!d) return -1;
        struct dirent *e; int r = 0;
        while (!r && (e = readdir(d))) {
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            size_t a = strlen(src) + strlen(e->d_name) + 2, b = strlen(dst) + strlen(e->d_name) + 2;
            char *s = malloc(a), *t = malloc(b); snprintf(s, a, "%s/%s", src, e->d_name); snprintf(t, b, "%s/%s", dst, e->d_name);
            r = clone_one(s, t); free(s); free(t);
        }
        closedir(d);
        return r ? r : copy_xattrs(src, dst);
    }
    if (S_ISLNK(st.st_mode)) { char l[4096]; ssize_t k = readlink(src, l, sizeof l - 1); if (k < 0) return -1; l[k] = 0; return symlink(l, dst); }
    int in = open(src, O_RDONLY | O_CLOEXEC); if (in < 0) return -1;
    int out = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, st.st_mode & 07777); if (out < 0) { close(in); return -1; }
    int r = 0;
    if (ioctl(out, FICLONE, in)) {
        char buf[1 << 16]; ssize_t k;
        while ((k = read(in, buf, sizeof buf)) > 0) for (ssize_t o = 0; o < k;) { ssize_t w = write(out, buf + o, (size_t)(k - o)); if (w < 0) { r = -1; break; } o += w; }
        if (k < 0) r = -1;
    }
    close(in); if (close(out)) r = -1;
    return r ? r : copy_xattrs(src, dst);
}
int clonefile(const char *src, const char *dst, uint32_t flags) {
    (void)flags; struct stat st;
    if (!lstat(dst, &st)) { errno = EEXIST; return -1; }
    return clone_one(src, dst);
}
int renamex_np(const char *from, const char *to, unsigned flags) {
    unsigned f = (flags & RENAME_SWAP) ? RENAME_EXCHANGE : (flags & RENAME_EXCL) ? RENAME_NOREPLACE : 0;
    return renameat2(AT_FDCWD, from, AT_FDCWD, to, f);
}

#undef confstr
size_t dm_confstr(int name, char *buf, size_t n) {
    if (name != _CS_DARWIN_USER_TEMP_DIR) return confstr(name, buf, n);
    struct stat st; const char *t = getenv("TMPDIR");
    const char *d = !stat("/dev/shm", &st) && S_ISDIR(st.st_mode) && !access("/dev/shm", W_OK) ? "/dev/shm" : t && *t ? t : "/tmp";
    int k = snprintf(buf, n, "%s/", d); return k < 0 ? 0 : (size_t)k + 1;
}
const char *dm_statfs_type(const struct statfs *fs) {
    switch ((unsigned long)fs->f_type) {
    case TMPFS_MAGIC: return "tmpfs"; case EXT4_SUPER_MAGIC: return "ext4"; case BTRFS_SUPER_MAGIC: return "btrfs"; case XFS_SUPER_MAGIC: return "xfs";
    case NFS_SUPER_MAGIC: return "nfs"; case 0x65735546: return "fuse"; case 0x5346544e: return "ntfs"; case 0x2011BAB0: return "exfat"; case 0xF2F52010: return "f2fs";
    default: return "other";
    }
}
