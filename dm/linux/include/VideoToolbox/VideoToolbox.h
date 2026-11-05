// VideoToolbox / CoreMedia / CoreVideo on Linux: types and constants only, every call fails. AV1 movies then decode with dav1d (VTIsHardwareDecodeSupported is false);
// AVC movies show black frames until a Linux H.264 decoder backs hle_movie_avc.c.
#pragma once
#include "../dm_apple_media.h"
typedef struct dm_vt *VTDecompressionSessionRef;
typedef struct dm_fd *CMVideoFormatDescriptionRef, *CMFormatDescriptionRef;
typedef struct dm_bb *CMBlockBufferRef;
typedef struct dm_sb *CMSampleBufferRef;
typedef struct dm_pb *CVPixelBufferRef, *CVImageBufferRef, *CVBufferRef;
typedef UInt32 VTDecodeInfoFlags, VTDecodeFrameFlags, CMVideoCodecType, CVPixelBufferLockFlags, OSType;
typedef struct { int64_t value; int32_t timescale; uint32_t flags; int64_t epoch; } CMTime;
typedef struct { CMTime duration, presentationTimeStamp, decodeTimeStamp; } CMSampleTimingInfo;
typedef struct { int32_t width, height; } CMVideoDimensions;
typedef void (*VTDecompressionOutputCallback)(void *, void *, OSStatus, VTDecodeInfoFlags, CVImageBufferRef, CMTime, CMTime);
typedef struct { VTDecompressionOutputCallback decompressionOutputCallback; void *decompressionOutputRefCon; } VTDecompressionOutputCallbackRecord;
static const CMTime kCMTimeInvalid = {0, 0, 0, 0};
static inline CMTime CMTimeMake(int64_t v, int32_t ts) { CMTime t = {v, ts, 1, 0}; return t; }
enum { kCMVideoCodecType_AV1 = 0x61763031, kCVPixelFormatType_32BGRA = 0x42475241, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange = 0x34323076, kCVPixelBufferLock_ReadOnly = 1 };
#define kCMFormatDescriptionExtension_YCbCrMatrix CFSTR("YCbCrMatrix")
#define kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms CFSTR("SampleDescriptionExtensionAtoms")
#define kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2 CFSTR("ITU_R_709_2")
#define kCVImageBufferYCbCrMatrixKey CFSTR("CVImageBufferYCbCrMatrix")
#define kCVImageBufferYCbCrMatrix_ITU_R_709_2 CFSTR("ITU_R_709_2")
#define kCVImageBufferYCbCrMatrix_ITU_R_2020 CFSTR("ITU_R_2020")
#define kCVPixelBufferPixelFormatTypeKey CFSTR("PixelFormatType")
#define VTIsHardwareDecodeSupported(...) (DM_IGN(__VA_ARGS__), (Boolean)0)
#define VTDecompressionSessionCreate(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define VTDecompressionSessionDecodeFrame(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define VTDecompressionSessionInvalidate(...) DM_IGN(__VA_ARGS__)
#define VTDecompressionSessionWaitForAsynchronousFrames(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CMVideoFormatDescriptionCreate(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CMVideoFormatDescriptionCreateFromH264ParameterSets(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CMVideoFormatDescriptionGetDimensions(...) (DM_IGN(__VA_ARGS__), (CMVideoDimensions){0, 0})
#define CMFormatDescriptionGetExtension(...) (DM_IGN(__VA_ARGS__), (CFTypeRef)0)
#define CMBlockBufferCreateWithMemoryBlock(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CMBlockBufferReplaceDataBytes(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CMSampleBufferCreateReady(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CVBufferCopyAttachment(...) (DM_IGN(__VA_ARGS__), (CFTypeRef)0)
#define CVPixelBufferLockBaseAddress(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define CVPixelBufferUnlockBaseAddress(...) (DM_IGN(__VA_ARGS__), noErr)
#define CVPixelBufferRetain(p) (p)
#define CVPixelBufferRelease(...) DM_IGN(__VA_ARGS__)
#define CVPixelBufferGetBaseAddress(...) (DM_IGN(__VA_ARGS__), (void *)0)
#define CVPixelBufferGetBaseAddressOfPlane(...) (DM_IGN(__VA_ARGS__), (void *)0)
#define CVPixelBufferGetBytesPerRow(...) (DM_IGN(__VA_ARGS__), (size_t)0)
#define CVPixelBufferGetBytesPerRowOfPlane(...) (DM_IGN(__VA_ARGS__), (size_t)0)
#define CVPixelBufferGetWidth(...) (DM_IGN(__VA_ARGS__), (size_t)0)
#define CVPixelBufferGetHeight(...) (DM_IGN(__VA_ARGS__), (size_t)0)
#define CVPixelBufferGetWidthOfPlane(...) (DM_IGN(__VA_ARGS__), (size_t)0)
#define CVPixelBufferGetHeightOfPlane(...) (DM_IGN(__VA_ARGS__), (size_t)0)
