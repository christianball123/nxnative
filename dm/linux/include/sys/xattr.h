// Apple's extended-attribute calls take a position and options (XATTR_NOFOLLOW); Linux's do not, and user attributes need the "user." namespace.
#pragma once
#include_next <sys/xattr.h>
#include "../dm_linux.h"
#ifndef XATTR_NOFOLLOW
#define XATTR_NOFOLLOW 1
#endif
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
ssize_t dm_getxattr(const char *p, const char *name, void *v, size_t n, uint32_t pos, int opt);
int dm_setxattr(const char *p, const char *name, const void *v, size_t n, uint32_t pos, int opt);
int dm_removexattr(const char *p, const char *name, int opt);
#define getxattr dm_getxattr
#define setxattr dm_setxattr
#define removexattr dm_removexattr
