#!/bin/sh
# Builds and runs the Vulkan replay test on this machine (x86_64 or arm64): needs clang, glslangValidator, a Vulkan loader + driver and the validation layer
# (Ubuntu: apt install glslang-tools mesa-vulkan-drivers vulkan-validationlayers libvulkan-dev; Fedora: dnf install glslang mesa-vulkan-drivers vulkan-validation-layers vulkan-loader-devel)
# Expected: "alpha test: row 50 col 40 ffff0000 (discarded: ffff0000), row 50 col 56 80202020 (drawn: 80202020)", "pixels: row 0 80202020, row 20 80202020, row 40 ffff0000, row 63 ffff0000; quad: row 50 col 5 80202020, row 50 col 50 ffff0000", "linear target's guest memory: row 0 80202020, row 50 col 5 80202020, row 50 col 50 ffff0000", "storage buffer after the submission: 7 8 9 10 ... 14", "buffer copy after the write-back in flight: 7 8 ... 14", "fence wait: 1 (1 expected), completed 1", "alias past another texture at the same storage: row 50 col 50 = the target", "validation errors: 0"
set -e
cd "$(dirname "$0")/../.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir "$T/spv" "$T/inc"; cp tests/vk/dm.h "$T/inc/"
glslangValidator -V tests/vk/t.vert -o "$T/spv/1111111111111111.spv" >/dev/null
glslangValidator -V tests/vk/t.frag -o "$T/spv/2222222222222222.spv" >/dev/null
glslangValidator -V tests/vk/t.comp -o "$T/spv/3333333333333333.spv" >/dev/null
glslangValidator -V tests/vk/t.geom -o "$T/spv/4444444444444444.spv" >/dev/null   # a geometry shader: the base translation, and the one for triangles (_g3) a draw picks
cp "$T/spv/4444444444444444.spv" "$T/spv/4444444444444444_g3.spv"
echo '{"cbufs":[],"sbufs":[],"textures":[],"images":[]}' > "$T/spv/4444444444444444.res.json"; cp "$T/spv/4444444444444444.res.json" "$T/spv/4444444444444444_g3.res.json"
echo '{"cbufs":[{"binding":1,"slot":3}],"sbufs":[],"textures":[],"images":[]}' > "$T/spv/1111111111111111.res.json"
echo '{"cbufs":[{"binding":2,"slot":3}],"sbufs":[],"textures":[{"binding":1,"type":3,"cbuf_slot":-1,"handle":8}],"images":[]}' > "$T/spv/2222222222222222.res.json"
echo '{"cbufs":[],"sbufs":[{"binding":1,"slot":0,"cb_slot":0,"cb_offset":196,"flags":1}],"textures":[],"images":[]}' > "$T/spv/3333333333333333.res.json"
python3 tools/spv_pack.py "$T/spv" "$T/t.dmsp" >/dev/null
GEN=$(ls -d build/*/gen 2>/dev/null | head -1); [ -n "$GEN" ] || { echo "build the host once first (make GAME=<id>): the test takes its game profile header"; exit 1; }
F="-O0 -g -fblocks -D_GNU_SOURCE -w -I $T/inc -I include -I ../host -I ../gfx/nvn/include -I ../third_party/vulkan-headers/include -I $GEN"
for s in gfx_vk_replay.c gfx_vk_shader.c ../gfx/tex/nvn_fmt_mtl.c ../gfx/nvn/nvn_tex.c ../gfx/tex/nvn_deswizzle.c tests/vk/replay_test.c; do clang $F -c $s -o "$T/$(basename $s .c).o"; done
clang -o "$T/t" "$T"/*.o -ldl -lpthread -lm
if [ -n "$DM_TEST_GDB" ]; then gdb -batch -ex run -ex bt --args "$T/t" 1111111111111111 2222222222222222 "$T/t.dmsp" 4444444444444444; else "$T/t" 1111111111111111 2222222222222222 "$T/t.dmsp" 4444444444444444 2>&1 | grep -v "missing vk"; fi
