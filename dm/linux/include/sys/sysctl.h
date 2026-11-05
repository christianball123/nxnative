// BSD sysctl on Linux: sysctlbyname answers the hw.* names the host reads; sysctl(CTL_KERN, KERN_PROC, KERN_PROC_PID) answers the process start time (main.c's boot timeline)
#pragma once
#include "../dm_linux.h"
#define CTL_KERN 1
#define KERN_PROC 14
#define KERN_PROC_PID 1
struct kinfo_proc { struct { struct timeval p_starttime; } kp_proc; };
int sysctl(int *mib, unsigned n, void *old, size_t *oldlen, void *new_, size_t newlen);
