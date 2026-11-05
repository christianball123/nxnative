// Linux (aarch64) stand-ins for the Apple APIs the host uses. The host's sources include Apple headers by name (mach/mach.h, dispatch/dispatch.h, ...);
// on Linux dm/linux/include is first on the include path, and each of those names is a small header that includes this one. The implementations are in
// dm/linux/compat.c. Only what the host calls is here; a call that has no Linux meaning (Mach thread sampling, power counters) fails cleanly, as it would
// on an Apple system that refused it, and the host already handles that failure.
#pragma once
#ifndef __linux__
#error "dm/linux/include is only for Linux builds"
#endif
#define _GNU_SOURCE 1
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/vfs.h>
#include <sys/statvfs.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- TargetConditionals ----
#ifndef TARGET_OS_IPHONE
#define TARGET_OS_IPHONE 0
#endif
#ifndef TARGET_OS_OSX
#define TARGET_OS_OSX 0
#endif
#ifndef TARGET_OS_MAC
#define TARGET_OS_MAC 0
#endif
#ifndef TARGET_OS_SIMULATOR
#define TARGET_OS_SIMULATOR 0
#endif
#define DM_LINUX 1

// ---- cache control (libkern/OSCacheControl.h) ----
static inline void sys_icache_invalidate(void *p, size_t n) { __builtin___clear_cache((char *)p, (char *)p + n); }
static inline void sys_dcache_flush(void *p, size_t n) { __builtin___clear_cache((char *)p, (char *)p + n); }

// ---- time (mach/mach_time.h) ----
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t, *mach_timebase_info_t;
uint64_t mach_absolute_time(void);   // nanoseconds of CLOCK_MONOTONIC: timebase 1/1
static inline uint64_t mach_continuous_time(void) { struct timespec t; clock_gettime(CLOCK_BOOTTIME, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static inline int mach_timebase_info(mach_timebase_info_data_t *i) { i->numer = 1; i->denom = 1; return 0; }
int mach_wait_until(uint64_t deadline);
#ifndef CLOCK_UPTIME_RAW
#define CLOCK_UPTIME_RAW CLOCK_MONOTONIC_RAW
#endif
#ifndef CLOCK_MONOTONIC_RAW_APPROX
#define CLOCK_MONOTONIC_RAW_APPROX CLOCK_MONOTONIC_COARSE
#endif
#ifndef CLOCK_UPTIME_RAW_APPROX
#define CLOCK_UPTIME_RAW_APPROX CLOCK_MONOTONIC_COARSE
#endif
static inline uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; if (clock_gettime(c, &t)) return 0; return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

// ---- Mach basics (mach/mach.h) ----
typedef int kern_return_t;
typedef int boolean_t;
typedef unsigned int natural_t;
typedef int integer_t;
typedef unsigned int mach_port_t;
typedef mach_port_t thread_act_t, task_t, thread_t;
typedef thread_act_t *thread_act_array_t;
typedef natural_t mach_msg_type_number_t;
typedef uintptr_t vm_address_t, vm_offset_t, vm_size_t, mach_vm_address_t, mach_vm_size_t;
typedef int vm_prot_t, vm_inherit_t;
typedef int *task_info_t, *thread_info_t, *thread_state_t, *thread_policy_t, *vm_region_info_t;
typedef int task_flavor_t, thread_flavor_t, vm_region_flavor_t, thread_policy_flavor_t;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
enum { KERN_SUCCESS = 0, KERN_INVALID_ADDRESS = 1, KERN_PROTECTION_FAILURE = 2, KERN_NO_SPACE = 3, KERN_INVALID_ARGUMENT = 4, KERN_FAILURE = 5, KERN_NOT_SUPPORTED = 46 };
enum { VM_PROT_NONE = 0, VM_PROT_READ = 1, VM_PROT_WRITE = 2, VM_PROT_EXECUTE = 4 };
#define VM_PROT_ALL (VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE)
#define VM_PROT_DEFAULT (VM_PROT_READ | VM_PROT_WRITE)
enum { VM_FLAGS_FIXED = 0, VM_FLAGS_ANYWHERE = 1, VM_FLAGS_OVERWRITE = 0x4000 };
enum { VM_INHERIT_SHARE = 0, VM_INHERIT_COPY = 1, VM_INHERIT_NONE = 2 };
extern uintptr_t vm_page_size;
#define MACH_PORT_NULL 0u
static inline mach_port_t mach_task_self(void) { return 1; }
mach_port_t mach_thread_self(void);
static inline kern_return_t mach_port_deallocate(mach_port_t t, mach_port_t p) { (void)t; (void)p; return KERN_SUCCESS; }
mach_port_t pthread_mach_thread_np(pthread_t t);

// Memory. vm_allocate gives shared-memory (memfd) pages, so vm_remap with copy = FALSE can map the same pages a second time, as Mach does.
// A region vm_remap cannot alias (memory not made by vm_allocate) is copied when copy = TRUE and refused otherwise.
kern_return_t vm_allocate(mach_port_t t, vm_address_t *addr, vm_size_t size, int flags);
kern_return_t vm_deallocate(mach_port_t t, vm_address_t addr, vm_size_t size);
kern_return_t vm_protect(mach_port_t t, vm_address_t addr, vm_size_t size, boolean_t set_max, vm_prot_t prot);
kern_return_t vm_remap(mach_port_t t, vm_address_t *to, vm_size_t size, vm_offset_t mask, int flags, mach_port_t src_task, vm_address_t from, boolean_t copy,
                       vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit);
kern_return_t vm_read_overwrite(mach_port_t t, vm_address_t from, vm_size_t size, vm_address_t to, vm_size_t *got);   // a read that fails instead of faulting
kern_return_t vm_write(mach_port_t t, vm_address_t to, vm_offset_t from, mach_msg_type_number_t size);
#define mach_vm_allocate vm_allocate
#define mach_vm_deallocate vm_deallocate
#define mach_vm_protect vm_protect
#define mach_vm_remap vm_remap
#define mach_vm_read_overwrite vm_read_overwrite
#define vm_region_64 mach_vm_region
#define VM_REGION_BASIC_INFO_64 9
typedef struct { vm_prot_t protection, max_protection; vm_inherit_t inheritance; boolean_t shared, reserved; uint64_t offset; int behavior; unsigned short user_wired_count; } vm_region_basic_info_data_64_t;
#define VM_REGION_BASIC_INFO_COUNT_64 ((mach_msg_type_number_t)(sizeof(vm_region_basic_info_data_64_t) / sizeof(int)))
kern_return_t mach_vm_region(mach_port_t t, mach_vm_address_t *addr, mach_vm_size_t *size, int flavor, vm_region_info_t info, mach_msg_type_number_t *cnt, mach_port_t *obj);

// Task and thread information: only what the host reads. task_info TASK_VM_INFO answers phys_footprint (resident set); the rest answer KERN_FAILURE
#define TASK_VM_INFO 22
#define TASK_POWER_INFO_V2 26
typedef struct { uint64_t virtual_size, region_count, page_size, resident_size, resident_size_peak, device, device_peak, internal, internal_peak, external, external_peak,
                 reusable, reusable_peak, purgeable_volatile_pmap, purgeable_volatile_resident, purgeable_volatile_virtual, compressed, compressed_peak, compressed_lifetime,
                 phys_footprint, min_address, max_address; } task_vm_info_data_t;
#define TASK_VM_INFO_COUNT ((mach_msg_type_number_t)(sizeof(task_vm_info_data_t) / sizeof(natural_t)))
typedef struct { uint64_t task_interrupt_wakeups, task_platform_idle_wakeups, task_timer_wakeups_bin_1, task_timer_wakeups_bin_2, total_user, total_system; } task_power_info_data_t;
typedef struct { uint64_t task_gpu_utilisation, task_gpu_stat_reserved0, task_gpu_stat_reserved1, task_gpu_stat_reserved2; } gpu_energy_data;
typedef struct { task_power_info_data_t cpu_energy; gpu_energy_data gpu_energy; uint64_t task_ptime, task_pset_switches, task_energy; } task_power_info_v2_data_t;
#define TASK_POWER_INFO_V2_COUNT ((mach_msg_type_number_t)(sizeof(task_power_info_v2_data_t) / sizeof(natural_t)))
kern_return_t task_info(mach_port_t t, int flavor, task_info_t info, mach_msg_type_number_t *cnt);
kern_return_t task_threads(mach_port_t t, thread_act_array_t *list, mach_msg_type_number_t *n);

typedef struct { int seconds, microseconds; } time_value_t;
#define THREAD_BASIC_INFO 3
#define THREAD_EXTENDED_INFO 5
enum { TH_STATE_RUNNING = 1, TH_STATE_STOPPED, TH_STATE_WAITING, TH_STATE_UNINTERRUPTIBLE, TH_STATE_HALTED };
typedef struct { time_value_t user_time, system_time; integer_t cpu_usage, policy, run_state, flags, suspend_count, sleep_time; } thread_basic_info_data_t;
#define THREAD_BASIC_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_basic_info_data_t) / sizeof(natural_t)))
typedef struct { uint64_t pth_user_time, pth_system_time; int32_t pth_cpu_usage, pth_policy, pth_run_state, pth_flags, pth_sleep_time, pth_curpri, pth_priority, pth_maxpriority; char pth_name[64]; } thread_extended_info_data_t;
#define THREAD_EXTENDED_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_extended_info_data_t) / sizeof(natural_t)))
kern_return_t thread_info(mach_port_t th, int flavor, thread_info_t info, mach_msg_type_number_t *cnt);
kern_return_t thread_suspend(mach_port_t th);
kern_return_t thread_resume(mach_port_t th);
#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT ((mach_msg_type_number_t)(sizeof(struct __darwin_arm_thread_state64) / sizeof(natural_t)))
kern_return_t thread_get_state(mach_port_t th, int flavor, thread_state_t st, mach_msg_type_number_t *cnt);
#define THREAD_TIME_CONSTRAINT_POLICY 2
typedef struct { uint32_t period, computation, constraint; boolean_t preemptible; } thread_time_constraint_policy_data_t;
#define THREAD_TIME_CONSTRAINT_POLICY_COUNT ((mach_msg_type_number_t)(sizeof(thread_time_constraint_policy_data_t) / sizeof(integer_t)))
kern_return_t thread_policy_set(mach_port_t th, int flavor, thread_policy_t p, mach_msg_type_number_t cnt);
kern_return_t thread_policy_get(mach_port_t th, int flavor, thread_policy_t p, mach_msg_type_number_t *cnt, boolean_t *get_default);

// ---- signal context: Apple's register-state structs laid over Linux's mcontext ----
// Linux's struct sigcontext holds regs[31], sp, pc, pstate in the same order as Darwin's __x[29], __fp, __lr, __sp, __pc, __cpsr, so the Darwin view is a cast;
// the NEON state and ESR live in records of __reserved, found by their magic. dm/host/dm.h's DM_UC_SS / DM_UC_NS / DM_UC_ESR pick the view on either system.
struct __darwin_arm_thread_state64 { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; };
struct __darwin_arm_neon_state64 { uint32_t __magic, __size, __fpsr, __fpcr; __uint128_t __v[32]; };   // the layout of Linux's struct fpsimd_context
#define _STRUCT_ARM_THREAD_STATE64 struct __darwin_arm_thread_state64
#define _STRUCT_ARM_NEON_STATE64 struct __darwin_arm_neon_state64
typedef struct __darwin_arm_thread_state64 arm_thread_state64_t;
#define __darwin_arm_thread_state64_get_pc(ts) ((ts).__pc)
#define __darwin_arm_thread_state64_get_lr(ts) ((ts).__lr)
#define __darwin_arm_thread_state64_get_sp(ts) ((ts).__sp)
#define __darwin_arm_thread_state64_get_fp(ts) ((ts).__fp)
#define __darwin_arm_thread_state64_set_pc_fptr(ts, p) ((ts).__pc = (uint64_t)(uintptr_t)(p))
#define __darwin_arm_thread_state64_set_lr_fptr(ts, p) ((ts).__lr = (uint64_t)(uintptr_t)(p))
#define __darwin_arm_thread_state64_set_sp(ts, p) ((ts).__sp = (uint64_t)(uintptr_t)(p))
#define __darwin_arm_thread_state64_set_fp(ts, p) ((ts).__fp = (uint64_t)(uintptr_t)(p))
static inline struct __darwin_arm_thread_state64 *dm_uc_ss(ucontext_t *uc) { return (struct __darwin_arm_thread_state64 *)(void *)&uc->uc_mcontext.regs[0]; }
static inline void *dm_uc_record(ucontext_t *uc, uint32_t magic) {
    uint8_t *p = (uint8_t *)uc->uc_mcontext.__reserved, *e = p + sizeof uc->uc_mcontext.__reserved;
    while (p + 8 <= e) { uint32_t m = ((uint32_t *)p)[0], sz = ((uint32_t *)p)[1]; if (!m || !sz) break; if (m == magic) return p; p += sz; }
    return NULL;
}
static inline struct __darwin_arm_neon_state64 *dm_uc_ns(ucontext_t *uc) {
    static __thread struct __darwin_arm_neon_state64 none; void *r = dm_uc_record(uc, 0x46508001);   // FPSIMD_MAGIC
    return r ? (struct __darwin_arm_neon_state64 *)r : &none;
}
static inline uint64_t dm_uc_esr(ucontext_t *uc) { uint64_t *r = (uint64_t *)dm_uc_record(uc, 0x45535201); return r ? r[1] : 0; }   // ESR_MAGIC: {head, esr}

// ---- threads and QoS (pthread/qos.h) ----
typedef enum { QOS_CLASS_USER_INTERACTIVE = 0x21, QOS_CLASS_USER_INITIATED = 0x19, QOS_CLASS_DEFAULT = 0x15, QOS_CLASS_UTILITY = 0x11, QOS_CLASS_BACKGROUND = 0x09, QOS_CLASS_UNSPECIFIED = 0 } qos_class_t;
int pthread_set_qos_class_self_np(qos_class_t q, int rel);
int pthread_get_qos_class_np(pthread_t t, qos_class_t *q, int *rel);
int pthread_attr_set_qos_class_np(pthread_attr_t *a, qos_class_t q, int rel);
typedef struct dm_qos_override *pthread_override_t;
static inline pthread_override_t pthread_override_qos_class_start_np(pthread_t t, qos_class_t q, int rel) { (void)t; (void)q; (void)rel; return (pthread_override_t)0; }
static inline int pthread_override_qos_class_end_np(pthread_override_t o) { (void)o; return 0; }
qos_class_t qos_class_self(void);
void *pthread_get_stackaddr_np(pthread_t t);   // the top of the stack, as on Apple systems
size_t pthread_get_stacksize_np(pthread_t t);
int pthread_cond_timedwait_relative_np(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel);
static inline void pthread_jit_write_protect_np(int on) { (void)on; }   // Linux keeps no per-thread W^X switch: JIT pages are mapped rwx
static inline int pthread_jit_write_protect_supported_np(void) { return 0; }
#ifndef MAP_JIT
#define MAP_JIT 0
#endif
int pthread_threadid_np(pthread_t t, uint64_t *id);
// Apple's pthread_setname_np names only the calling thread
#define pthread_setname_np(name) dm_pthread_setname_self(name)
int dm_pthread_setname_self(const char *name);

// ---- malloc (malloc/malloc.h) ----
static inline size_t malloc_size(const void *p) { return malloc_usable_size((void *)p); }
static inline size_t malloc_good_size(size_t n) { return n; }
typedef struct dm_malloc_zone malloc_zone_t;
static inline malloc_zone_t *malloc_default_zone(void) { return (malloc_zone_t *)0; }
static inline void *malloc_zone_malloc(malloc_zone_t *z, size_t n) { (void)z; return malloc(n); }
static inline void *malloc_zone_calloc(malloc_zone_t *z, size_t c, size_t n) { (void)z; return calloc(c, n); }
static inline void *malloc_zone_realloc(malloc_zone_t *z, void *p, size_t n) { (void)z; return realloc(p, n); }
static inline void *malloc_zone_memalign(malloc_zone_t *z, size_t a, size_t n) { (void)z; void *p = NULL; return posix_memalign(&p, a < sizeof(void *) ? sizeof(void *) : a, n) ? NULL : p; }
static inline void malloc_zone_free(malloc_zone_t *z, void *p) { (void)z; free(p); }
static inline malloc_zone_t *malloc_create_zone(size_t start, unsigned flags) { (void)start; (void)flags; return (malloc_zone_t *)(uintptr_t)1; }   // one heap: glibc's
static inline void malloc_set_zone_name(malloc_zone_t *z, const char *n) { (void)z; (void)n; }
static inline void malloc_destroy_zone(malloc_zone_t *z) { (void)z; }
static inline size_t malloc_zone_size_dm(malloc_zone_t *z, const void *p) { (void)z; return malloc_usable_size((void *)p); }

// ---- libc differences ----
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));   // stable, as BSD's
#ifndef F_GETPATH
#define F_GETPATH 50   // handled by fcntl's wrapper below
#endif
int dm_fcntl(int fd, int cmd, ...);
#define fcntl dm_fcntl
#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif
#ifndef F_FULLFSYNC
#define F_FULLFSYNC 51
#endif
#ifndef MAP_NOCACHE
#define MAP_NOCACHE 0
#endif
#ifndef O_EVTONLY
#define O_EVTONLY O_RDONLY
#endif
#define RUSAGE_INFO_V4 4
struct rusage_info_v4 { uint8_t ri_uuid[16]; uint64_t ri_user_time, ri_system_time, ri_pkg_idle_wkups, ri_interrupt_wkups, ri_pageins, ri_wired_size, ri_resident_size, ri_phys_footprint,
                        ri_proc_start_abstime, ri_proc_exit_abstime, ri_child_user_time, ri_child_system_time, ri_child_pkg_idle_wkups, ri_child_interrupt_wkups, ri_child_pageins,
                        ri_child_elapsed_abstime, ri_diskio_bytesread, ri_diskio_byteswritten, ri_cpu_time_qos_default, ri_cpu_time_qos_maintenance, ri_cpu_time_qos_background,
                        ri_cpu_time_qos_utility, ri_cpu_time_qos_legacy, ri_cpu_time_qos_user_initiated, ri_cpu_time_qos_user_interactive, ri_billed_system_time, ri_serviced_system_time,
                        ri_logical_writes, ri_lifetime_max_phys_footprint, ri_instructions, ri_cycles, ri_billed_energy, ri_serviced_energy, ri_interval_max_phys_footprint, ri_runnable_time; };
typedef void *rusage_info_t;
int proc_pid_rusage(int pid, int flavor, rusage_info_t *buffer);
// struct stat's Darwin names for the timestamps
#define st_mtimespec st_mtim
#define st_atimespec st_atim
#define st_ctimespec st_ctim
#define st_birthtimespec st_ctim
// The per-user temporary folder. Linux: /dev/shm when present (tmpfs, which is what the guest-memory files want: shared, sparse, in RAM), else $TMPDIR or /tmp
#define _CS_DARWIN_USER_TEMP_DIR 0x10001
size_t dm_confstr(int name, char *buf, size_t n);
#define confstr dm_confstr
#ifndef MNT_LOCAL
#define MNT_LOCAL 0x1000
#endif
const char *dm_statfs_type(const struct statfs *fs);   // Darwin's f_fstypename: "tmpfs", "ext4", "btrfs", ...   // reads /proc/self/io and status: disk bytes, logical writes, peak footprint
int sysctlbyname(const char *name, void *old, size_t *oldlen, void *new_, size_t newlen);

// ---- Mach-O sections (mach-o/getsect.h) ----
// On ELF, the registration tables (__DATA,swhle / swforce / swnvn) are sections of their own name, bounded by the linker's __start_ / __stop_ symbols.
// The game modules, which the Mac and iPhone builds link into the executable, are files on Linux: getsectiondata finds __dm<name>x0 / __DM<name>W,__img /
// __dm<name>x1 / __dm<name>x2 in <DM_MODULES>/<name>.patched.bin through its fixups.tsv, as dm/ios/tools/gen_embed.py cuts them, and maps that piece.
struct mach_header_64 { uint32_t magic; };
extern const struct mach_header_64 _mh_execute_header;
uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *seg, const char *sect, unsigned long *size);
#define DM_SECTION(seg, sect) section(#sect)

// ---- CommonCrypto (CommonCrypto/CommonDigest.h): SHA-256 only ----
#define CC_SHA256_DIGEST_LENGTH 32
#define CC_SHA256_BLOCK_BYTES 64
typedef uint32_t CC_LONG;
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; uint32_t n; } CC_SHA256_CTX;
int CC_SHA256_Init(CC_SHA256_CTX *c);
int CC_SHA256_Update(CC_SHA256_CTX *c, const void *data, CC_LONG len);
int CC_SHA256_Final(unsigned char *md, CC_SHA256_CTX *c);
unsigned char *CC_SHA256(const void *data, CC_LONG len, unsigned char *md);

#ifdef __cplusplus
}
#endif
