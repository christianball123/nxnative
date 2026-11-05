# The host for Linux aarch64 (ArmadaOS on the AYN Thor, any arm64 Linux with glibc). The same sources as dm/host/Makefile, without the Apple-only Objective-C files:
# dm/linux/include stands in for the Apple headers (dm_linux.h), dm/linux/compat.c implements them, and dm/linux/*.c replace the .m files.
#   make -C dm/linux GAME=<base game title ID>            native build on an arm64 machine
#   make -C dm/linux GAME=<id> CROSS=1                    cross build from x86_64 (clang --target=aarch64-linux-gnu, Debian/Ubuntu's arm64 cross libc)
# Output: dm/linux/build/<id>/dmhost
ifeq ($(strip $(GAME)),)
ifneq ($(filter-out clean,$(or $(MAKECMDGOALS),dmhost)),)
$(error GAME is not set: make GAME=<base game title ID>)
endif
endif
H = ../host
G = ../gfx
T = ../third_party
B = build/$(GAME)
GAME_DIR := $(if $(and $(wildcard ../games/$(GAME)/game.tsv),$(if $(DM_GAMES_FIRST),,1)),../games/$(GAME),$(shell python3 ../games/game.py link $(GAME) $(B)/profile >&2 && echo $(B)/profile))
GAME_H = $(B)/gen/game_profile.h

CC = clang
ifeq ($(CROSS),1)
TARGET = --target=aarch64-linux-gnu
LDFLAGS_X = -fuse-ld=lld
endif
OPT ?= -O1
CFLAGS = $(TARGET) $(OPT) -g -Wall -Wno-unused-function -Wno-format -fblocks -D_GNU_SOURCE -march=armv8.2-a+lse -fno-omit-frame-pointer -ffixed-x18 -pthread
# -ffixed-x18: the rewritten game code keeps the guest's x18 in a thread slot, but host code called from it must not rely on x18 either (it stays the platform register, as on Apple systems)
INC = -I include -I $(H) -I $(G)/nvn/include -I $(T)/vulkan-headers/include -I $(dir $(GAME_H)) -I $(T)/libvpx -I $(T)/dav1d/include -I $(T)/stb_vorbis -I $(H)/gen/msl-macosx

HOST_C = $(H)/main.c $(H)/loader.c $(H)/hle.c $(H)/crash.c $(H)/probe.c $(H)/thread.c $(H)/save_sync.c $(sort $(wildcard $(H)/hle_*.c))
GFX_C = $(sort $(wildcard $(G)/*.c $(G)/nvn/*.c $(G)/tex/*.c))
LINUX_C = compat.c $(sort $(wildcard host_*.c gfx_*.c))
ASM = $(H)/entry.S $(G)/nvn/nvn_entry.S $(G)/nvn/nvn_thunks.S
ASM_ELF = $(patsubst %.S,$(B)/asm/%.S,$(notdir $(ASM)))
OBJ = $(patsubst %.c,$(B)/obj/%.o,$(notdir $(HOST_C) $(GFX_C) $(LINUX_C))) $(ASM_ELF:.S=.o)
vpath %.c $(H) $(G) $(G)/nvn $(G)/tex .

# Third party: libvpx (VP8/VP9 movies) and dav1d (AV1 movies), as dm/host/Makefile builds them; dav1d with dm/linux/third_party/dav1d/config.h (ELF symbols)
VPX = $(T)/libvpx
VPX_SRC = $(sort $(shell find $(VPX) -name '*.c'))
VPX_OBJ = $(patsubst $(VPX)/%.c,$(B)/vpx/%.o,$(VPX_SRC))
DAV1D = $(T)/dav1d
DAV1D_SRC = $(sort $(filter-out %_tmpl.c,$(wildcard $(DAV1D)/src/*.c)) $(DAV1D)/src/arm/cpu.c $(wildcard $(DAV1D)/config/tmpl/*.c) $(filter-out %_tmpl.S %/util.S,$(wildcard $(DAV1D)/src/arm/64/*.S)))
DAV1D_OBJ = $(patsubst $(DAV1D)/%,$(B)/dav1d/%.o,$(DAV1D_SRC))

.DEFAULT_GOAL := dmhost
dmhost: $(B)/dmhost
$(GAME_H): $(GAME_DIR)/game.tsv ../games/game.py
	@mkdir -p $(dir $@); python3 ../games/game.py header $(GAME) $@
$(B)/asm/%.S: $(H)/%.S tools/asm_elf.py
	@mkdir -p $(dir $@); python3 tools/asm_elf.py $< $@
$(B)/asm/%.S: $(G)/nvn/%.S tools/asm_elf.py
	@mkdir -p $(dir $@); python3 tools/asm_elf.py $< $@
$(B)/asm/%.o: $(B)/asm/%.S
	$(CC) $(TARGET) -c $< -o $@
$(B)/obj/%.o: %.c $(GAME_H) $(wildcard include/*.h include/*/*.h) $(H)/dm.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
$(B)/vpx/%.o: $(VPX)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TARGET) -O2 -g -fPIC -I$(VPX) -I$(VPX)/config -c $< -o $@
$(B)/vpx/libvpx.a: $(VPX_OBJ)
	rm -f $@; ar rcs $@ $^
$(B)/dav1d/%.o: $(DAV1D)/%
	@mkdir -p $(dir $@)
	$(CC) $(TARGET) -O2 -g -fvisibility=hidden -fomit-frame-pointer -I third_party/dav1d -I$(DAV1D) -I$(DAV1D)/include -I$(DAV1D)/config -c $< -o $@
$(B)/dav1d/libdav1d.a: $(DAV1D_OBJ)
	rm -f $@; ar rcs $@ $^
$(B)/dmhost: $(OBJ) $(B)/vpx/libvpx.a $(B)/dav1d/libdav1d.a
	$(CC) $(TARGET) $(LDFLAGS_X) -pthread -rdynamic -o $@ $(OBJ) $(B)/vpx/libvpx.a $(B)/dav1d/libdav1d.a -lm -ldl $(LIBS)
clean:
	rm -rf build
.PHONY: dmhost clean
