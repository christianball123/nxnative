// The little of CoreFoundation the host's C files use: a pointer-keyed mutable dictionary (hle_nvn.c) and the bundle lookup main.c makes for the commit
// (Linux builds have no bundle: it answers NULL, and main.c prints "not recorded").
#pragma once
#include "../dm_linux.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef const void *CFTypeRef;
typedef const struct dm_cf_string *CFStringRef;
typedef struct dm_cf_dict *CFMutableDictionaryRef;
typedef const struct dm_cf_dict *CFDictionaryRef;
typedef const struct dm_cf_bundle *CFBundleRef;
typedef const void *CFAllocatorRef;
typedef long CFIndex;
typedef unsigned long CFTypeID;
typedef uint32_t CFStringEncoding;
#define kCFStringEncodingUTF8 0x08000100u
#define CFSTR(s) ((CFStringRef)(s))
typedef const struct dm_cf_number *CFNumberRef;
typedef const struct dm_cf_data *CFDataRef;
typedef const struct dm_cf_array *CFArrayRef;
typedef int CFNumberType;
enum { kCFNumberSInt32Type = 3 };
typedef struct { long version; void *retain, *release, *copyDescription, *equal, *hash; } CFDictionaryKeyCallBacks, CFDictionaryValueCallBacks;
extern const CFDictionaryKeyCallBacks kCFTypeDictionaryKeyCallBacks;
extern const CFDictionaryValueCallBacks kCFTypeDictionaryValueCallBacks;
// Only the movie code's VideoToolbox attributes build these, and VideoToolbox refuses every call on Linux: they hold nothing
static inline CFNumberRef CFNumberCreate(CFAllocatorRef a, CFNumberType t, const void *v) { (void)a; (void)t; (void)v; return (CFNumberRef)1; }
static inline CFDataRef CFDataCreate(CFAllocatorRef a, const uint8_t *b, CFIndex n) { (void)a; (void)b; (void)n; return (CFDataRef)1; }
static inline CFDictionaryRef CFDictionaryCreate(CFAllocatorRef a, const void **k, const void **v, CFIndex n, const void *kc, const void *vc) { (void)a; (void)k; (void)v; (void)n; (void)kc; (void)vc; return (CFDictionaryRef)1; }
static inline int CFEqual(CFTypeRef a, CFTypeRef b) { return a == b; }
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *keycb, const void *valcb);
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *key);
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *value);
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key);
CFIndex CFDictionaryGetCount(CFDictionaryRef d);
void CFRelease(CFTypeRef o);
CFTypeRef CFRetain(CFTypeRef o);
static inline CFBundleRef CFBundleGetMainBundle(void) { return (CFBundleRef)0; }
static inline CFTypeRef CFBundleGetValueForInfoDictionaryKey(CFBundleRef b, CFStringRef k) { (void)b; (void)k; return (CFTypeRef)0; }
static inline CFTypeID CFGetTypeID(CFTypeRef o) { (void)o; return 0; }
static inline CFTypeID CFStringGetTypeID(void) { return 1; }
static inline int CFStringGetCString(CFStringRef s, char *buf, CFIndex n, CFStringEncoding e) { (void)s; (void)buf; (void)n; (void)e; return 0; }
#ifdef __cplusplus
}
#endif
