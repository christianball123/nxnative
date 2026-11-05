// AudioToolbox on Linux: types and constants only, every call fails. hle_audio.c then uses its silent sink (real-time pacing, no sound; the Mac default too, DM_AUDIO=1 asks
// for the device), and hle_movie.c's AAC decoder gives no audio. A real Linux output (SDL3) replaces au_* in hle_audio.c later.
#pragma once
#include "../dm_apple_media.h"
typedef struct dm_au *AudioUnit, *AudioComponentInstance;
typedef struct dm_ac *AudioComponent;
typedef struct dm_conv *AudioConverterRef;
typedef UInt32 AudioUnitPropertyID, AudioUnitScope, AudioUnitElement, AudioUnitRenderActionFlags, AudioFormatID, AudioFormatFlags, AudioDeviceID, AudioObjectID;
typedef struct { UInt32 componentType, componentSubType, componentManufacturer, componentFlags, componentFlagsMask; } AudioComponentDescription;
typedef struct { Float64 mSampleRate; AudioFormatID mFormatID; AudioFormatFlags mFormatFlags; UInt32 mBytesPerPacket, mFramesPerPacket, mBytesPerFrame, mChannelsPerFrame, mBitsPerChannel, mReserved; } AudioStreamBasicDescription;
typedef struct { SInt64 mStartOffset; UInt32 mVariableFramesInPacket, mDataByteSize; } AudioStreamPacketDescription;
typedef struct { UInt32 mNumberChannels, mDataByteSize; void *mData; } AudioBuffer;
typedef struct { UInt32 mNumberBuffers; AudioBuffer mBuffers[1]; } AudioBufferList;
typedef struct { Float64 mSampleTime; UInt64 mHostTime; Float64 mRateScalar; UInt64 mWordClockTime; UInt32 mFlags, mReserved; } AudioTimeStamp;
typedef struct { Float64 mMinimum, mMaximum; } AudioValueRange;
typedef OSStatus (*AURenderCallback)(void *, AudioUnitRenderActionFlags *, const AudioTimeStamp *, UInt32, UInt32, AudioBufferList *);
typedef struct { AURenderCallback inputProc; void *inputProcRefCon; } AURenderCallbackStruct;
typedef void (*AudioUnitPropertyListenerProc)(void *, AudioUnit, AudioUnitPropertyID, AudioUnitScope, AudioUnitElement);
typedef OSStatus (*AudioConverterComplexInputDataProc)(AudioConverterRef, UInt32 *, AudioBufferList *, AudioStreamPacketDescription **, void *);
enum { kAudioUnitType_Output = 0x61756f75, kAudioUnitSubType_DefaultOutput = 0x64656620, kAudioUnitSubType_RemoteIO = 0x72696f63, kAudioUnitManufacturer_Apple = 0x6170706c };
enum { kAudioUnitScope_Global = 0, kAudioUnitScope_Input = 1, kAudioUnitScope_Output = 2 };
enum { kAudioUnitProperty_StreamFormat = 8, kAudioUnitProperty_SetRenderCallback = 23, kAudioOutputUnitProperty_CurrentDevice = 2000,
       kAudioDevicePropertyBufferFrameSize = 0x6673697a, kAudioDevicePropertyBufferFrameSizeRange = 0x66737a23, kAudioConverterDecompressionMagicCookie = 0x646d6763 };
enum { kAudioFormatLinearPCM = 0x6c70636d, kAudioFormatMPEG4AAC = 0x61616320, kLinearPCMFormatFlagIsSignedInteger = 4, kLinearPCMFormatFlagIsPacked = 8, kAudioObjectUnknown = 0 };
#define AudioComponentFindNext(...) (DM_IGN(__VA_ARGS__), (AudioComponent)0)
#define AudioComponentInstanceNew(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioComponentInstanceDispose(...) (DM_IGN(__VA_ARGS__), noErr)
#define AudioUnitInitialize(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioUnitUninitialize(...) (DM_IGN(__VA_ARGS__), noErr)
#define AudioUnitGetProperty(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioUnitSetProperty(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioUnitAddPropertyListener(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioUnitRemovePropertyListenerWithUserData(...) (DM_IGN(__VA_ARGS__), noErr)
#define AudioOutputUnitStart(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioOutputUnitStop(...) (DM_IGN(__VA_ARGS__), noErr)
#define AudioConverterNew(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioConverterDispose(...) (DM_IGN(__VA_ARGS__), noErr)
#define AudioConverterSetProperty(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
#define AudioConverterFillComplexBuffer(...) (DM_IGN(__VA_ARGS__), DM_UNSUPPORTED)
