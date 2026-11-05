// Types shared by the AudioToolbox and VideoToolbox stand-ins.
#pragma once
#include "dm_linux.h"
#include "CoreFoundation/CoreFoundation.h"
typedef int32_t OSStatus;
typedef uint8_t UInt8; typedef uint16_t UInt16; typedef uint32_t UInt32; typedef uint64_t UInt64; typedef int32_t SInt32; typedef int64_t SInt64;
typedef double Float64; typedef float Float32; typedef unsigned char Boolean;
#define noErr 0
#define DM_UNSUPPORTED ((OSStatus)-12782)   // kVTCouldNotFindVideoDecoderErr / "unimplemented": every call below fails, and the callers fall back (silent sink, black movie frames, dav1d)
#define DM_IGN(...) ((void)sizeof((__VA_ARGS__), 0))
