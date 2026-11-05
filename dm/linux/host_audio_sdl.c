// The AudioUnit output interface hle_audio.c uses (include/AudioToolbox/AudioToolbox.h) on SDL3 audio. SDL3 is opened at run time like the window (gfx_vk_window.c), so a
// machine without it still runs: Start then fails, and hle_audio.c's silent sink stands in.
// Pushed, not pulled: the game queues about 20 ms of sound in 5 ms buffers (4 in flight), and a sound server that asks for a bigger period at once (PipeWire's default is
// 1024 frames, 21 ms, and it ignores a smaller request from SDL) finds too little queued and fills the rest with silence: the sound stutters (seen on the AYN Thor). So a feeder
// thread asks the unit's render callback (hle_audio.c au_render) for 5 ms every 5 ms and keeps the SDL stream about DM_AUDIO_CUSHION_MS ahead (default 50 ms), so the
// server's period no longer has to fit in what the game has queued.
#include "dm.h"
#include <AudioToolbox/AudioToolbox.h>
#include <dlfcn.h>
#include <stdatomic.h>

typedef struct SDL_AudioStream SDL_AudioStream;
typedef struct { int format, channels, freq; } SDL_AudioSpec;
typedef void (*SDL_AudioStreamCallback)(void *userdata, SDL_AudioStream *stream, int additional, int total);
enum { SDL_INIT_AUDIO = 0x10, SDL_AUDIO_S16LE = 0x8010 };
#define SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK 0xFFFFFFFFu
static struct {
    void *lib; int tried, ok;
    bool (*InitSubSystem)(uint32_t);
    const char *(*GetError)(void);
    bool (*SetHint)(const char *, const char *);
    SDL_AudioStream *(*OpenAudioDeviceStream)(uint32_t, const SDL_AudioSpec *, SDL_AudioStreamCallback, void *);
    bool (*PutAudioStreamData)(SDL_AudioStream *, const void *, int);
    int (*GetAudioStreamQueued)(SDL_AudioStream *);
    uint32_t (*GetAudioStreamDevice)(SDL_AudioStream *);
    bool (*GetAudioDeviceFormat)(uint32_t, SDL_AudioSpec *, int *);
    bool (*ResumeAudioStreamDevice)(SDL_AudioStream *);
    bool (*PauseAudioStreamDevice)(SDL_AudioStream *);
    void (*DestroyAudioStream)(SDL_AudioStream *);
    const char *(*GetCurrentAudioDriver)(void);
} A;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int sdl_audio(void) {
    pthread_mutex_lock(&g_mu);
    if (!A.tried) {
        A.tried = 1;
        A.lib = dlopen("libSDL3.so.0", RTLD_NOW | RTLD_LOCAL); if (!A.lib) A.lib = dlopen("libSDL3.so", RTLD_NOW | RTLD_LOCAL);
        if (!A.lib) LOG("[RED] audio: SDL3 is not installed (sudo dnf install -y SDL3 in the toolbox): no sound");
        else {
            int ok = 1;
#define L(f) if (!(*(void **)&A.f = dlsym(A.lib, "SDL_" #f))) { LOG("[RED] audio: SDL3 has no SDL_" #f); ok = 0; }
            L(InitSubSystem) L(GetError) L(SetHint) L(OpenAudioDeviceStream) L(PutAudioStreamData) L(GetAudioStreamQueued) L(GetAudioStreamDevice) L(GetAudioDeviceFormat) L(ResumeAudioStreamDevice) L(PauseAudioStreamDevice) L(DestroyAudioStream) L(GetCurrentAudioDriver)
#undef L
            if (ok) {
                // a short device period (SDL asks PipeWire for it as the node's latency); the cushion follows the period the device really has (feeder)
                const char *fr = getenv("DM_AUDIO_FRAMES"); A.SetHint("SDL_AUDIO_DEVICE_SAMPLE_FRAMES", fr && *fr ? fr : "256");
                A.SetHint("SDL_AUDIO_DEVICE_APP_NAME", "nxnative");
                if (!A.InitSubSystem(SDL_INIT_AUDIO)) LOG("[RED] audio: SDL audio does not start: %s", A.GetError());
                else { A.ok = 1; LOG("audio: SDL3 audio, driver %s", A.GetCurrentAudioDriver() ? A.GetCurrentAudioDriver() : "?"); }
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return A.ok;
}

struct dm_au { AudioStreamBasicDescription fmt; AURenderCallbackStruct cb; SDL_AudioStream *st; _Atomic int running; pthread_t th; uint8_t buf[48000 / 200 * 8 * 4]; };
struct dm_ac { int unused; };
static struct dm_ac g_comp;

static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void *feeder(void *p) {
    AudioUnit u = p; dm_pthread_setname_self("audio feeder");
    UInt32 bpf = u->fmt.mBytesPerFrame ? u->fmt.mBytesPerFrame : 4, frames = (UInt32)(u->fmt.mSampleRate / 200), n = frames * bpf;   // 5 ms
    if (n > sizeof u->buf) { frames = (UInt32)(sizeof u->buf / bpf); n = frames * bpf; }
    // The cushion: DM_AUDIO_CUSHION_MS, else two of the device's periods and at least 40 ms. A sound server may run a far longer period than asked (a handheld saving power:
    // PipeWire at 4096 or 8192 frames), and a cushion shorter than one period leaves a gap at every period: the sound pauses and resumes (seen on the AYN Thor at 50 ms)
    SDL_AudioSpec ds = {0}; int dev_frames = 0;
    if (!A.GetAudioDeviceFormat(A.GetAudioStreamDevice(u->st), &ds, &dev_frames) || dev_frames <= 0) dev_frames = 1024;
    const char *c = getenv("DM_AUDIO_CUSHION_MS");
    int ms = c && *c ? atoi(c) : (int)(2.0 * dev_frames * 1000 / (ds.freq > 0 ? ds.freq : 48000)) + 10;
    if (ms < 40) ms = 40;
    int cushion = (int)(u->fmt.mSampleRate * ms / 1000) * (int)bpf;
    LOG("audio: device period %d frames at %d Hz: cushion %d ms", dev_frames, ds.freq, ms);
    // The cushion starts as silence, so the game is never asked for a burst: its audio renderer is paced by its buffers' release (hle_audio.c, hle_audren.c), and releasing
    // several at once put it out of step with the game's updates (catch-up frames, underruns: 115 in 2000 periods on the Thor with a fill-to-cushion feeder)
    memset(u->buf, 0, n);
    for (int k = 0; k < cushion; k += (int)n) A.PutAudioStreamData(u->st, u->buf, (int)n);
    // Then one period every period of real time, as the silent sink does; the device's clock and CLOCK_MONOTONIC differ by a few ppm, so the period is trimmed by up to 0.5 %
    // toward a queue of one cushion (the game's pace follows; nothing is dropped or padded for it)
    const uint64_t period = (uint64_t)frames * 1000000000ull / (uint64_t)u->fmt.mSampleRate;
    uint64_t next = mono_ns();
    while (atomic_load(&u->running)) {
        AudioBufferList bl = {1, {{u->fmt.mChannelsPerFrame, n, u->buf}}};
        AudioUnitRenderActionFlags fl = 0; AudioTimeStamp ts = {0};
        if (u->cb.inputProc(u->cb.inputProcRefCon, &fl, &ts, 0, frames, &bl)) memset(u->buf, 0, n);
        A.PutAudioStreamData(u->st, u->buf, (int)n);
        int q = A.GetAudioStreamQueued(u->st);
        if (q < (int)n && cushion < (int)(u->fmt.mSampleRate * bpf)) {   // ran dry: the device takes more per period than the cushion holds; grow it (up to 1 s), once with silence
            int add = cushion / 2; cushion += add; memset(u->buf, 0, n);
            for (int k = 0; k < add; k += (int)n) A.PutAudioStreamData(u->st, u->buf, (int)n);
            LOG("[RED] audio: the device ran dry: cushion raised to %d ms", (int)((int64_t)cushion * 1000 / ((int64_t)u->fmt.mSampleRate * bpf)));
            q = A.GetAudioStreamQueued(u->st);
        }
        int64_t trim = q > cushion * 3 / 2 ? (int64_t)period / 200 : q < cushion / 2 ? -(int64_t)period / 200 : 0;
        next += period + trim;
        uint64_t now = mono_ns(); if (next + 50000000 < now) next = now;   // asleep for 50 ms or more (a stall): start from now instead of catching up in a burst
        struct timespec t = {(time_t)(next / 1000000000ull), (long)(next % 1000000000ull)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
    }
    return NULL;
}

AudioComponent AudioComponentFindNext(AudioComponent after, const AudioComponentDescription *d) { (void)d; return after ? NULL : &g_comp; }
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *out) { (void)c; sdl_audio(); *out = calloc(1, sizeof **out); return *out ? noErr : -1; }   // SDL starts here: a slow start inside Start tripped hle_audio.c's 200 ms watchdog
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u) { if (!u) return noErr; AudioOutputUnitStop(u); if (u->st) A.DestroyAudioStream(u->st); free(u); return noErr; }
OSStatus AudioUnitInitialize(AudioUnit u) { (void)u; return noErr; }
OSStatus AudioUnitUninitialize(AudioUnit u) { (void)u; return noErr; }
// No device properties: hle_audio.c au_buffer reads the current device first and leaves the buffer size alone when there is none (SDL takes it from DM_AUDIO_FRAMES)
OSStatus AudioUnitGetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement el, void *data, UInt32 *size) {
    (void)scope; (void)el;
    if (id == kAudioUnitProperty_StreamFormat && *size >= sizeof u->fmt) { memcpy(data, &u->fmt, sizeof u->fmt); *size = sizeof u->fmt; return noErr; }
    return DM_UNSUPPORTED;
}
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement el, const void *data, UInt32 size) {
    (void)scope; (void)el;
    if (id == kAudioUnitProperty_StreamFormat && size >= sizeof u->fmt) { memcpy(&u->fmt, data, sizeof u->fmt); return noErr; }
    if (id == kAudioUnitProperty_SetRenderCallback && size >= sizeof u->cb) { memcpy(&u->cb, data, sizeof u->cb); return noErr; }
    return DM_UNSUPPORTED;
}
OSStatus AudioUnitAddPropertyListener(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc proc, void *data) { (void)u; (void)id; (void)proc; (void)data; return noErr; }   // SDL follows device changes itself
OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit u, AudioUnitPropertyID id, AudioUnitPropertyListenerProc proc, void *data) { (void)u; (void)id; (void)proc; (void)data; return noErr; }
OSStatus AudioOutputUnitStart(AudioUnit u) {
    if (!u || !u->cb.inputProc) return -1;
    if (u->running) return noErr;
    if (!sdl_audio()) return -1;
    if (!u->st) {
        if (u->fmt.mBitsPerChannel != 16 || !(u->fmt.mFormatFlags & kLinearPCMFormatFlagIsSignedInteger)) { LOG("[RED] audio: output format %u-bit flags %#x is not 16-bit signed", u->fmt.mBitsPerChannel, u->fmt.mFormatFlags); return -1; }
        SDL_AudioSpec sp = {SDL_AUDIO_S16LE, (int)u->fmt.mChannelsPerFrame, (int)u->fmt.mSampleRate};
        u->st = A.OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &sp, NULL, NULL);
        if (!u->st) { LOG("[RED] audio: SDL cannot open the output (%d channels, %d Hz): %s", sp.channels, sp.freq, A.GetError()); return -1; }
        LOG("audio: SDL output %d channels, %d Hz", sp.channels, sp.freq);
    }
    atomic_store(&u->running, 1);
    if (pthread_create(&u->th, NULL, feeder, u)) { atomic_store(&u->running, 0); return -1; }
    if (!A.ResumeAudioStreamDevice(u->st)) { LOG("[RED] audio: SDL output does not start: %s", A.GetError()); AudioOutputUnitStop(u); return -1; }
    return noErr;
}
OSStatus AudioOutputUnitStop(AudioUnit u) {
    if (u && u->st && atomic_exchange(&u->running, 0)) { pthread_join(u->th, NULL); A.PauseAudioStreamDevice(u->st); }
    return noErr;
}
