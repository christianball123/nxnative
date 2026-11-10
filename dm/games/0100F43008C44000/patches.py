# Pokémon Legends: Z-A (0100F43008C44000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a module's NSO build ID;
# CUTS, OBS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
from nso_asm import ADRP, B

# 60 fps (30 on the Switch), switchable: the profile's fps_target (0 here: the game's own 30) or DM_FPS_TARGET=60 at launch.
# The game's frame clock reads its frame mode, "vsyncs per frame - 1" (1 on the Switch: 30 fps), with `ldrsb w20, [x9, #0x20]`; it sets the present interval from it and steps
# its clock channels by max(that, the vsyncs that really passed). Located by ZA Recomp (Hexveil404/PokeZ-ARecomp, MIT, tools/za_patches.py: ZA_BUILDS, the 2.0.2 site
# 0x7001b296d8 = main+0x1b296d8; its 1.0.0 site is the community's static 60 fps patch, `14008052` at 019E37A8). The stub gives the mode from the host's frame step instead
# (state page +0x20, decided by hle_nvn.c fstep_tick from how long frames stay on screen): step 1 → mode 0 (60 fps, a present every vsync), step 2 or more → mode 1 (30 fps, as on
# the Switch; the game itself catches up past that with the vsyncs that passed). So when 60 does not hold the game falls back to its own 30 instead of running in slow motion.
# Target 0 (off) or no step decided yet: the original load. Only w20 is used: the load overwrites it, so its old value is dead at the site; no flags are touched (cbz, sub,
# neg and lsr leave them), as min(step - 1, 1) is computed without a compare (neg, then the sign bit)
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    SITE = 0x1b296d8
    fs_site(SITE, '3481c039')                                        # ldrsb w20,[x9,#0x20]   the frame mode
    start = here()                                                   # words: 0 adrp, 1 ldr, 2 cbz, 3 adrp, 4 ldr, 5 cbz, 6 sub, 7 neg, 8 lsr, 9 b, 10 orig ldrsb, 11 b
    CBZ_W20 = lambda words: 0x34000000 | (words & 0x7ffff) << 5 | 20 # cbz w20, +words
    emit(ADRP(20, here(), FS_STATE_OFF)); emit(0xb9400294)          # ldr w20,[x20]          target fps (float bits; 0 = off)
    emit(CBZ_W20(10 - 2))                                            # cbz w20,orig
    emit(ADRP(20, here(), FS_STATE_OFF)); emit(0xb9402294)          # ldr w20,[x20,#0x20]    the host's step in vsyncs
    emit(CBZ_W20(10 - 5))                                            # cbz w20,orig           not decided yet
    emit(0x51000694)                                                 # sub w20,w20,#1         vsyncs per frame - 1
    emit(0x4b1403f4)                                                 # neg w20,w20
    emit(0x531f7e94)                                                 # lsr w20,w20,#31        min(it, 1): 0 = 60 fps, 1 = 30 fps
    emit(B(here(), SITE + 4))                                        # back
    assert here() == start + 40, 'frame mode stub layout'
    emit(0x39c08134); emit(B(here(), SITE + 4))                      # orig: ldrsb w20,[x9,#0x20]; back
    fs_put(SITE, [B(SITE, start)], 'frame mode: from the host\'s frame step (state page target fps; 0 = the game\'s own 30), 60 fps where it holds')

FRAME_STEP = {'b1f12fd919eae86ab8a978317677e64b': frame_step}

# How general each patch is and could be (dm/tools/docs_patches.py "Binding", listed in docs/agents/game-patches.md): what, its binding now, the loosest it could have (goal),
# purpose, why; blocked or kept: why the gap stays open (cannot close yet, or left open on purpose); checked: what was checked on game files (else a judgement)
BINDING = [
    {'what': 'frame mode from the host\'s frame step (60 fps, switchable)', 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'why': "the game's own frame-mode load; ZA Recomp lists the same site for 1.0.0 and 2.0.0 (0x19e37a8, 0x1b29cf8)",
     'blocked': 'frame-step stubs have no function-matched form yet'},
]
