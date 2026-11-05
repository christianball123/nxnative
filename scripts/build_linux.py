#!/usr/bin/env python3
# Build a game for Linux aarch64 (ArmadaOS on the AYN Thor, or any arm64 Linux) and write a script that runs it.
# Stage 1 of the Linux port: the null graphics device (dm/linux/gfx_null.c), so the game runs headless, with no picture and no sound. What it shows is whether the
# game's own code boots and reaches its frame loop on Linux: the log prints "[null gfx] present #1" when the game hands over its first frame.
#
# The steps are those of "Build by hand" in docs/development.md, without the shaders (no GPU yet) and without Xcode:
#   1. unpack the game: a .nup file (scripts/pack_nup.py) or a dump folder (scripts/pack_dump.py) → <work>/extract/v<version>/ExeFS + RomFS.img
#   2. rewrite the code modules for Linux (dm/tools/patch_nso.py with DM_TARGET=linux) → <work>/linux/patched_v<version>
#   3. build the host (make -C dm/linux GAME=<title ID>) and copy it beside them
#   4. write <work>/linux/run.sh with the environment the host reads (DM_ROMFS, DM_SAVE, DM_TEMP, DM_ROOT)
#
# Usage:
#   python3 scripts/build_linux.py ~/Games/MarioTennisFever.nup
#   python3 scripts/build_linux.py ~/Games/MarioTennisFever            (a dump folder: base/exefs, base/romfs, updates/...)
#   python3 scripts/build_linux.py <game> --version 0 --work-dir ~/Games/MTF_build
# Then run the script it prints, and send the log (<work>/linux/log.txt).
#
# Needs: python3 (3.14, or pip install backports.zstd), capstone 5 (pip install 'capstone>=5,<6'), clang, lld, make. On ArmadaOS do this inside a toolbox:
#   toolbox create nxnative && toolbox enter nxnative && sudo dnf install -y clang lld make python3-pip git && pip install --user 'capstone>=5,<6'
import argparse, os, platform, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))


def say(s): print(s, flush=True)
def die(s): sys.exit(f'[RED] {s}')


def run(cmd, env=None, cwd=None):
    say('+ ' + ' '.join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd)
    if r.returncode: die(f'{Path(str(cmd[1] if Path(str(cmd[0])).name.startswith("python") else cmd[0])).name} failed (exit {r.returncode})')


def check_tools():
    try:
        import capstone
        if capstone.cs_version()[0] != 5: die(f'capstone {capstone.__version__} found; the rewriter needs 5.x: {sys.executable} -m pip install --user "capstone>=5,<6"')
    except ImportError: die(f'capstone is not installed: {sys.executable} -m pip install --user "capstone>=5,<6"')
    try: from compression import zstd   # noqa: F401
    except ImportError:
        try: from backports import zstd   # noqa: F401
        except ImportError: die(f'no zstd module: use Python 3.14, or {sys.executable} -m pip install --user backports.zstd')
    for t in ('clang', 'make'):
        if not shutil.which(t): die(f'{t} is not installed (inside the toolbox: sudo dnf install -y clang lld make)')


def unpack(game, version, work):
    import pack_nup, pack_dump
    if game.is_file() and pack_nup.is_nup(game) or game.is_dir() and any(p.suffix.lower() == '.nup' for p in game.iterdir()):
        nup = game if game.is_file() else next(p for p in game.iterdir() if p.suffix.lower() == '.nup')
        with pack_nup.Nup(nup) as n:
            app, base, vers, dlc = pack_nup.scan(n)
            ver = version if version is not None else max(vers, default=0)
            if ver and ver not in vers: die(f'no update {ver} in {nup.name} (have: {", ".join(map(str, sorted(vers))) or "none"})')
            k = vers[ver] if ver else base
            if k is None: die(f'{nup.name}: no base game (version 0); pick an update with --version')
            work = work or nup.parent / f'{app:016X}_build'
            out = work / 'extract' / f'v{ver}'
            say(f'{app:016X} version {ver}: unpacking {k.name}')
            pack_nup.pack_version(n, k, base, app, out, say)
            pack_nup.pack_dlc(n, app, dlc, work / 'aoc', say)
            return f'{app:016X}', ver, work, out
    if game.is_dir():
        cmd = [sys.executable, REPO / 'scripts/pack_dump.py', game] + (['--version', str(version)] if version is not None else []) + (['--work-dir', work] if work else [])
        say('+ ' + ' '.join(map(str, cmd)))
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
        if r.returncode: die('pack_dump.py failed')
        line = next((l for l in r.stdout.splitlines() if l.startswith('done: ')), None)
        if not line: die('pack_dump.py did not say where it wrote the game')
        out = Path(line[6:].split(',')[0]).resolve()
        work = out.parent.parent
        ver = int(out.name[1:])
        head = next((l for l in r.stdout.splitlines() if len(l) > 16 and l[16:25] == ' version ' and all(c in '0123456789ABCDEF' for c in l[:16])), '')
        if not head: die('pack_dump.py did not print the title ID')
        return head[:16], ver, work, out
    die(f'{game}: neither a .nup file nor a dump folder')


def main():
    ap = argparse.ArgumentParser(description='Build a Switch / Switch 2 game for Linux aarch64 (null graphics: headless boot test)')
    ap.add_argument('game', type=Path, help='a .nup file, a folder holding one, or a dump folder')
    ap.add_argument('--version', type=int, help='the update to build (default: the newest; 0 = the base game)')
    ap.add_argument('--work-dir', type=Path, help='the build folder (default: <title ID>_build beside the game)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if platform.machine() not in ('aarch64', 'arm64'): say(f'[RED] this machine is {platform.machine()}: the build runs here, but the game only runs on arm64 Linux')
    check_tools()
    tid, ver, work, ext = unpack(a.game.expanduser().resolve(), a.version, a.work_dir.expanduser().resolve() if a.work_dir else None)
    exefs = ext / 'ExeFS'
    if not exefs.is_dir(): die(f'no ExeFS in {ext}')
    # The profile: the repository's, or for a game nxnative has none for, a provisional one guessed from the files (the survey the NSP script makes)
    prof = REPO / 'dm/games' / tid
    env = dict(os.environ, DM_TARGET='linux')
    if not (prof / 'game.tsv').is_file():
        pdir = work / 'profile'
        if not (pdir / tid / 'game.tsv').is_file():
            # The NSP script surveys such a game and guesses a provisional profile (dm/games/profile_guess.py); that step still runs only on the Mac
            die(f'{tid}: nxnative has no profile for this game yet. Put one in {pdir / tid} (game.tsv), or send the code modules in {exefs} to get one made')
        env['DM_GAMES_DIR'] = str(pdir)
    mods = subprocess.run([sys.executable, REPO / 'dm/games/game.py', tid, 'modules'], capture_output=True, text=True, env=env).stdout.split()
    if not mods: mods = [p.name for p in exefs.iterdir() if p.name in ('main', 'sdk') or p.name.startswith('subsdk')]
    lw = work / 'linux'; patched = lw / f'patched_v{ver}'
    patched.mkdir(parents=True, exist_ok=True)
    for m in mods:
        if not (exefs / m).is_file(): say(f'[RED] {m}: not in {exefs}, skipped'); continue
        run([sys.executable, REPO / 'dm/tools/patch_nso.py', exefs / m, patched, m], env=dict(env, DM_GAME=tid))
    run(['make', '-C', REPO / 'dm/linux', f'GAME={tid}', f'-j{a.jobs}'] + ([] if platform.machine() in ('aarch64', 'arm64') else ['CROSS=1']), env=env)
    host = lw / 'dmhost'
    shutil.copy2(REPO / 'dm/linux/build' / tid / 'dmhost', host)
    for d in ('save', 'temp'): (lw / 'run' / d).mkdir(parents=True, exist_ok=True)
    aoc = work / 'aoc'
    script = lw / 'run.sh'
    script.write_text(f'''#!/bin/sh
# Runs {tid} v{ver} with the null graphics device (headless). The log goes to the screen and to log.txt beside this script.
# Ctrl+C stops it. DM_TIME_LIMIT=<seconds> stops it by itself.
cd "$(dirname "$0")"
export DM_ROOT="{lw}/run" DM_ROMFS="{ext / 'RomFS.img'}" DM_SAVE="{lw}/run/save" DM_TEMP="{lw}/run/temp"
{f'export DM_AOC_DIR="{aoc}"' if aoc.is_dir() and any(aoc.iterdir()) else '# no DLC'}
exec ./dmhost "{patched}" "$@" 2>&1 | tee log.txt
''')
    script.chmod(0o755)
    say(f'\ndone. Run it with:\n  {script}\nthen send me {lw / "log.txt"}')


if __name__ == '__main__':
    main()
