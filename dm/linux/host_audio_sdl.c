// The AudioUnit output interface hle_audio.c uses (include/AudioToolbox/AudioToolbox.h) on SDL3 audio: one SDL audio stream per unit, whose callback asks the unit's render
// callback (hle_audio.c au_render) for exactly the bytes SDL needs, as CoreAudio does. SDL3 is opened at run time like the window (gfx_vk_window.c), so a machine without it
// still runs: Start then fails, and hle_audio.c's silent sink stands in.
// DM_AUDIO_FRAMES=<n> sets the device buffer (default 240 frames, 5 ms at 48 kHz, as hle_audio.c asks of CoreAudio).
#include "dm.h"
#include <AudioToolbox/AudioToolbox.h>
#include <dlfcn.h>

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
            L(InitSubSystem) L(GetError) L(SetHint) L(OpenAudioDeviceStream) L(PutAudioStreamData) L(ResumeAudioStreamDevice) L(PauseAudioStreamDevice) L(DestroyAudioStream) L(GetCurrentAudioDriver)
#undef L
            if (ok) {
                const char *fr = getenv("DM_AUDIO_FRAMES"); A.SetHint("SDL_AUDIO_DEVICE_SAMPLE_FRAMES", fr && *fr ? fr : "240");
                A.SetHint("SDL_AUDIO_DEVICE_APP_NAME", "nxnative");
                if (!A.InitSubSystem(SDL_INIT_AUDIO)) LOG("[RED] audio: SDL audio does not start: %s", A.GetError());
                else { A.ok = 1; LOG("audio: SDL3 audio, driver %s", A.GetCurrentAudioDriver() ? A.GetCurrentAudioDriver() : "?"); }
            }
        }
    }
    pthread_mutex_unlock(&g_mu);
    return A.ok;
}

struct dm_au { AudioStreamBasicDescription fmt; AURenderCallbackStruct cb; SDL_AudioStream *st; int running; uint8_t *buf; size_t cap; };
struct dm_ac { int unused; };
static struct dm_ac g_comp;

static void feed(void *ud, SDL_AudioStream *st, int additional, int total) {
    (void)total;
    AudioUnit u = ud; if (additional <= 0) return;
    UInt32 bpf = u->fmt.mBytesPerFrame ? u->fmt.mBytesPerFrame : 4, frames = (UInt32)((additional + bpf - 1) / bpf), n = frames * bpf;
    if (u->cap < n) { u->buf = realloc(u->buf, n); u->cap = n; }
    AudioBufferList bl = {1, {{u->fmt.mChannelsPerFrame, n, u->buf}}};
    AudioUnitRenderActionFlags fl = 0; AudioTimeStamp ts = {0};
    if (u->cb.inputProc(u->cb.inputProcRefCon, &fl, &ts, 0, frames, &bl)) memset(u->buf, 0, n);
    A.PutAudioStreamData(st, u->buf, (int)n);
}

AudioComponent AudioComponentFindNext(AudioComponent after, const AudioComponentDescription *d) { (void)d; return after ? NULL : &g_comp; }
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *out) { (void)c; *out = calloc(1, sizeof **out); return *out ? noErr : -1; }
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u) { if (!u) return noErr; AudioOutputUnitStop(u); if (u->st) A.DestroyAudioStream(u->st); free(u->buf); free(u); return noErr; }
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
        u->st = A.OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &sp, feed, u);
        if (!u->st) { LOG("[RED] audio: SDL cannot open the output (%d channels, %d Hz): %s", sp.channels, sp.freq, A.GetError()); return -1; }
        LOG("audio: SDL output %d channels, %d Hz", sp.channels, sp.freq);
    }
    if (!A.ResumeAudioStreamDevice(u->st)) { LOG("[RED] audio: SDL output does not start: %s", A.GetError()); return -1; }
    u->running = 1; return noErr;
}
OSStatus AudioOutputUnitStop(AudioUnit u) { if (u && u->st && u->running) { A.PauseAudioStreamDevice(u->st); u->running = 0; } return noErr; }
