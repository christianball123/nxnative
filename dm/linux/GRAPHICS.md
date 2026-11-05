# Graphics on Linux: NVN on Vulkan

The Apple host replays NVN on Metal (`dm/gfx/nvn/nvn_mtl.m`, 13,700 lines) behind the C interface in `dm/gfx/nvn/nvn_mtl.h` and the `mtl_*` functions `hle_nvn.c` calls. The Linux host implements the same interface on Vulkan, in stages. `gfx_null.c` is that interface with no GPU, and each stage takes over part of it.

## What carries over and what does not

| Piece | On Linux |
|---|---|
| `hle_nvn.c`: the NVN interface, objects in the game's memory, command recording | as it is |
| Command record format (`nvn_obj.h`), texture layouts and deswizzling (`nvn_tex.c`, `dm/gfx/tex`) | as it is |
| Shader translation: Ryujinx's translator (`dmsh_xlate`, .NET) → SPIR-V | as it is. The SPIR-V is what Vulkan takes; the Metal path's later steps (spirv-cross → MSL → metallib) drop out |
| `dmsh_build.py`'s post-processing (depth rewrite `main0_z`, vertex padding `vs_pad`, binding reflection) | redone on SPIR-V: the depth rewrite as a vertex-stage `gl_Position.z` fix-up, the padding as extra outputs, reflection from the SPIR-V decorations |
| Replay (`nvn_mtl.m`) | rewritten for Vulkan (`gfx_vk_*.c`) |
| Memory pools: zero-copy `MTLBuffer` over guest memory | `VK_EXT_external_memory_host` (the guest pages imported as `VkDeviceMemory`) when the driver has it; else device memory mirrored into the guest addresses (a shared mapping `mremap`ped over the pool), else copies at submit |

## Stages

| Stage | What it gives | State |
|---|---|---|
| A | SDL3 window, Vulkan instance / device / swapchain, presents paced with the game's (`DM_WINDOW=1`) | in progress |
| B | memory pools on the GPU, textures (`VkImage`) with their storage, texture and buffer copies, clears; the game's swapchain image shown in the window | |
| C | shaders: SPIR-V per key, pipelines, draws (vertex streams, index buffers, uniform buffers, textures and samplers) | |
| D | the rest of Mario Tennis Fever's frame: depth / stencil, blending, compute, queries, the formats it uses | |
| E | controller (SDL3 gamepad → `hle_hid.c`), audio out (SDL3) | |
| F | performance: pipeline cache, prewarm lists, Adreno specifics | |

Mario Tennis Fever is the test game: it plays on the Mac, so a difference is the Linux layer's.
