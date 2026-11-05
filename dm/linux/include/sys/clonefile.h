// clonefile (APFS copy-on-write clone of a file or a whole directory tree) on Linux: a recursive copy that tries a reflink (FICLONE) per file first
// (btrfs and XFS clone; ext4 copies), keeping the guest-name extended attributes. renamex_np(RENAME_SWAP) is renameat2(RENAME_EXCHANGE).
#pragma once
#include "../dm_linux.h"
#define CLONE_NOFOLLOW 1
#define CLONE_NOOWNERCOPY 2
int clonefile(const char *src, const char *dst, uint32_t flags);
#ifndef RENAME_SWAP
#define RENAME_SWAP 2
#define RENAME_EXCL 4
#endif
int renamex_np(const char *from, const char *to, unsigned flags);
