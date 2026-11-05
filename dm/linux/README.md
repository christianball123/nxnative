# nxnative on Linux aarch64

A port of the nxnative host to arm64 Linux, aimed at ArmadaOS on the AYN Thor. The game's own ARM64 code runs directly on the CPU, as on Apple chips; what changes is the platform underneath.

## Status

| Part | State |
|---|---|
| Host (HLE services, threads, memory, files, saves) | builds for Linux aarch64 (`make GAME=<id>`), starts; untested with a game |
| Rewriter (`dm/tools/patch_nso.py`) | `DM_TARGET=linux`: patch code reads its thread slots through `tpidr_el0` |
| Graphics | null device (`gfx_null.c`); with `DM_WINDOW=1` a window (SDL3 + Vulkan, `gfx_vk_window.c`) presents with every game frame, no picture yet. Plan: [GRAPHICS.md](GRAPHICS.md) |
| Window, controller | window with `DM_WINDOW=1` (`DM_FULLSCREEN=1`); controller next |
| Audio | the host's silent sink (real-time pacing). Next: SDL3 output |
| Movies | AV1 through dav1d; AVC shows black frames (no H.264 decoder yet) |

## Build and run a game (stage 1: headless boot test)

On the Thor, inside a toolbox (ArmadaOS is read-only):

```
toolbox create nxnative && toolbox enter nxnative
sudo dnf install -y clang lld make git python3-pip
pip install --user 'capstone>=5,<6' lz4 zbic
git clone -b linux-arm64 https://github.com/christianball123/nxnative && cd nxnative
python3 scripts/build_linux.py ~/Games/MarioTennisFever.nup
```

It unpacks the game, rewrites its code for Linux, builds the host and prints the run script. The run log shows how far the game gets; `[null gfx] present #1` means it reached its frame loop.

## How the port is laid out

- `include/`: headers named like the Apple ones the host includes (`mach/mach.h`, `dispatch/dispatch.h`, `CommonCrypto/CommonDigest.h`, ...). Each includes `dm_linux.h`. The host's sources are unchanged apart from a few portable macros in `dm/host/dm.h` (`DM_UC_SS`, `DM_SECTION`, `DM_ASM_*`).
- `compat.c`: their implementations: Mach memory calls on `mmap` / `mremap`, thread information from `/proc`, a small libdispatch and blocks runtime, SHA-256, `clonefile` as a recursive copy, `RENAME_SWAP` as `renameat2(RENAME_EXCHANGE)`, xattrs in the `user.` namespace.
- `gfx_null.c`: the interface `nvn_mtl.m` gives the NVN layer, with no GPU.
- `tools/asm_elf.py`: the host's Mach-O assembly in ELF spelling.
- `third_party/dav1d/config.h`: dav1d's configuration with ELF symbols.

### Thread slots

The rewriter replaces the game's `x18`, `tpidr_el0` and `tpidrro_el0` with four per-thread slots. On Apple systems they are pthread TSD slots at `tpidrro_el0 + key * 8`. On Linux the patch code reads `tpidr_el0` (glibc's thread pointer), and the slots are the values of four pthread keys, which glibc keeps in a fixed array of `struct pthread`, at a fixed distance from the thread pointer. `loader.c` (`key_offsets`) finds that distance at start-up and writes it into each module's key page.

### Memory mirrors

Virtual memory pools and the cheats' writable view map the same pages twice. Linux does this with `mremap(old, 0, ...)` on shared mappings. `vm_allocate` therefore hands out shared anonymous memory, and the guest-memory files (`DM_GUEST_FILE`) go to `/dev/shm`.
