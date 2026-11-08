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
#   toolbox create nxnative && toolbox enter nxnative && sudo dnf install -y clang lld make python3-pip git && pip install --user 'capstone>=5,<6' lz4 zbic
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
    try: import lz4.block   # noqa: F401  (the code modules' LZ4 segments)
    except ImportError: die(f'lz4 is not installed: {sys.executable} -m pip install --user lz4 zbic')
    for t in ('clang', 'make'):
        if not shutil.which(t): die(f'{t} is not installed (inside the toolbox: sudo dnf install -y clang lld make)')


def unpack(game, version, work):
    import pack_nup, pack_dump
    if game.is_file() and pack_nup.is_nup(game) or game.is_dir() and any(p.suffix.lower() == '.nup' for p in game.iterdir()):
        if pack_nup.zstd is None: die(f'reading a .nup needs the zstd module: use Python 3.14, or {sys.executable} -m pip install --user backports.zstd')
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


def shaders(tid, ver, ext, exefs, nro_dir, lw, env, jobs, retranslate=False):
    # Graphics stage C: the game's shaders split out of its files (dm/gfx/tools/shader_split.py, as the NSP script does) and translated to SPIR-V (dm/linux/tools/spv_build.py)
    if not shutil.which('dotnet'): die('the shader step needs the .NET 10 runtime: sudo dnf install -y dotnet-runtime-10.0 (in the toolbox), then build again')
    sh = lw / f'shaders_v{ver}'; pkg, tables, spv = sh / 'pkg', sh / 'tables', sh / 'spv'
    env = dict(env, J=str(jobs))
    if nro_dir.is_dir(): env['DM_NRO_DIR'] = str(nro_dir)
    if not (sh / 'split.done').is_file():
        shutil.rmtree(pkg, ignore_errors=True); tables.mkdir(parents=True, exist_ok=True)
        say('splitting out the shaders (a few minutes)')
        mains = [exefs / 'main']
        run([sys.executable, REPO / 'dm/gfx/tools/shader_split.py', ext / 'RomFS.img', ','.join(map(str, mains)), pkg] +
            [tables / n for n in ('shader_map.tsv', 'shader_facts.tsv', 'shader_pairs.tsv', 'pso_pairs.tsv')], env=env)
        (sh / 'split.done').write_text('')
    else: say('shaders: already split')
    def translated():
        try: return any(e.name.endswith('.res.json') for e in os.scandir(spv))
        except OSError: return False
    if not retranslate and ((sh / 'spv.done').is_file() or translated()): say('shaders: already translated (--retranslate does it again)')
    else:
        say('translating the shaders to SPIR-V (can take an hour on the Thor; it prints nothing while it works)')
        run([sys.executable, REPO / 'dm/linux/tools/spv_build.py', pkg, spv], env=env)
        (sh / 'spv.done').write_text('')
    if not (sh / 'spv.done').is_file(): (sh / 'spv.done').write_text('')
    pack = lw / 'shaders.dmsp'
    packer = REPO / 'dm/linux/tools/spv_pack.py'
    if not pack.is_file() or pack.stat().st_mtime < max((sh / 'spv.done').stat().st_mtime, packer.stat().st_mtime):
        say('packing the shaders into one file (a few minutes)')
        run([sys.executable, REPO / 'dm/linux/tools/spv_pack.py', spv, pack])
    else: say(f'shaders: already packed ({pack})')

def main():
    ap = argparse.ArgumentParser(description='Build a Switch / Switch 2 game for Linux aarch64 (null graphics: headless boot test)')
    ap.add_argument('game', type=Path, help='a .nup file, a folder holding one, or a dump folder')
    ap.add_argument('--version', type=int, help='the update to build (default: the newest; 0 = the base game)')
    ap.add_argument('--work-dir', type=Path, help='the build folder (default: <title ID>_build beside the game)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--retranslate', action='store_true', help='translate the shaders again (after a translator change)')
    ap.add_argument('--shaders', action='store_true', help='also translate the game\'s shaders to SPIR-V (graphics stage C; needs the .NET 10 runtime, takes a while)')
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
    lw = work / 'linux'; patched = lw / f'patched_v{ver}'; nro_dir = lw / f'nro_v{ver}'
    patched.mkdir(parents=True, exist_ok=True)
    renv = dict(env, DM_GAME=tid)
    def rewrite(names):
        for m, src in names.items():
            if not src.is_file(): say(f'[RED] {m}: not in {src.parent}, skipped'); continue
            if (patched / f'{m}.patched.bin').is_file() and (patched / f'{m}.patched.bin').stat().st_mtime > src.stat().st_mtime: say(f'{m}: already rewritten'); continue
            run([sys.executable, REPO / 'dm/tools/patch_nso.py', src, patched, m], env=renv)
    rewrite({m: exefs / m for m in mods})
    # The NROs the game loads at run time with nn::ro (dm/tools/nro_collect.py): found in its files, or, when no file holds one (Mario Tennis Fever decodes its game code in memory),
    # caught from a first boot (DM_NRO_CAPTURE, dm/host/hle_ro.c), as the NSP script's capture_nro does
    romfs = ext / 'RomFS.img'
    nrrs = True
    if not (nro_dir / 'nro.tsv').is_file():
        tmp = nro_dir.with_name(nro_dir.name + '.partial'); shutil.rmtree(tmp, ignore_errors=True)
        r = subprocess.run([sys.executable, REPO / 'dm/tools/nro_collect.py', romfs, tmp, '--missing-ok'], capture_output=True, text=True, env=env)
        sys.stdout.write(r.stdout[-2000:])
        if r.returncode and 'no NRR file with hashes' in (r.stderr + r.stdout): nrrs = False; r = None; say('no NROs registered (the game loads none at run time)')   # Ocarina of Time
        elif r.returncode: sys.stderr.write(r.stderr[-2000:]); die('nro_collect.py failed')
        if r is not None and (tmp / 'nro.tsv').is_file(): shutil.rmtree(nro_dir, ignore_errors=True); tmp.rename(nro_dir)
        else: nrrs = False; shutil.rmtree(tmp, ignore_errors=True)
    host_built = False
    def build_host():
        nonlocal host_built
        if host_built: return
        run(['make', '-C', REPO / 'dm/linux', f'GAME={tid}', f'-j{a.jobs}'] + ([] if platform.machine() in ('aarch64', 'arm64') else ['CROSS=1']), env=env)
        shutil.copy2(REPO / 'dm/linux/build' / tid / 'dmhost', lw / 'dmhost'); host_built = True
    # Saves and the game's temporary files live on the internal storage: their files carry their guest name in an extended attribute and a save commits by an atomic
    # directory swap (renameat2 RENAME_EXCHANGE), which exFAT / FAT SD cards and USB drives do not support (the game aborts when its save cannot be created)
    run_dir = Path(os.environ.get('DM_RUN_DIR') or Path.home() / '.local/share/nxnative' / tid)
    for d in ('save', 'temp'): (run_dir / d).mkdir(parents=True, exist_ok=True)
    say(f'saves: {run_dir / "save"}')
    missing = nro_dir / 'missing.tsv'
    want = dict(l.split('\t')[:2] for l in missing.read_text().splitlines() if l) if missing.is_file() else {}
    if want and platform.machine() in ('aarch64', 'arm64'):
        build_host()
        cap = nro_dir / 'capture'; shutil.rmtree(cap, ignore_errors=True); cap.mkdir(parents=True)
        say(f'{len(want)} NRO(s) the game registers are in no file: booting it once to catch what it decodes (up to 2 minutes)')
        henv = dict(env, DM_ROMFS=str(romfs), DM_SAVE=str(run_dir / 'save'), DM_TEMP=str(run_dir / 'temp'), DM_ROOT=str(run_dir), DM_NRO_CAPTURE=str(cap))
        if (patched / 'nro.tsv').exists(): (patched / 'nro.tsv').unlink()
        with open(lw / 'nro_capture.log', 'w') as logf:
            p = subprocess.Popen([str(lw / 'dmhost'), str(patched)], env=henv, stdout=logf, stderr=subprocess.STDOUT)
            import time
            t0 = time.monotonic()
            while p.poll() is None and time.monotonic() - t0 < 120 and not any(cap.glob('*.nro')): time.sleep(0.5)
            if p.poll() is None: p.terminate(); p.wait()
        got = sorted(cap.glob('*.nro'))
        if not got or got[0].stem not in want:
            say(''.join(open(lw / 'nro_capture.log', errors='replace').readlines()[-30:]))
            die(f'the game decoded none of the NROs it registers (log: {lw / "nro_capture.log"})')
        rows = [l for l in (nro_dir / 'nro.tsv').read_text().splitlines() if l]
        name = f'nro{sum(1 for l in rows if not l.startswith(("#", "refused")))}'
        sha = got[0].stem
        shutil.move(got[0], nro_dir / f'{name}.nro')
        (nro_dir / 'nro.tsv').write_text('\n'.join(rows + [f'{name}\t{sha}\t{(nro_dir / f"{name}.nro").stat().st_size}\t{want.pop(sha)}\tcapture']) + '\n')
        missing.write_text(''.join(f'{h}\t{n}\n' for h, n in want.items()))
        shutil.rmtree(cap, ignore_errors=True)
        say(f'caught {name} ({(nro_dir / f"{name}.nro").stat().st_size / 2**20:.1f} MB)')
    elif want:
        say(f'[RED] {len(want)} NRO(s) can only be caught by booting the game, which needs arm64: run this on the Thor')
    if nrrs and (nro_dir / 'nro.tsv').is_file():
        nros = {l.split('\t')[0]: nro_dir / (l.split('\t')[0] + '.nro') for l in (nro_dir / 'nro.tsv').read_text().splitlines() if l and not l.startswith(('#', 'refused\t'))}
        rewrite(nros)
        shutil.copyfile(nro_dir / 'nro.tsv', patched / 'nro.tsv')
    build_host()
    # Fonts: the game asks the system for its shared fonts (dm/host/hle_sdk16.c). The standard one is any Unicode TTF on this machine (the Mac takes Arial Unicode);
    # the Nintendo extension (button icons) is generated by dm/tools/make_ext_font.py, which needs fontTools
    font = lw / 'font'; font.mkdir(exist_ok=True)
    if not (font / 'standard.ttf').exists():
        cands = ['NotoSans-Regular.ttf', 'NotoSansCJK-Regular.ttc', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf', 'Cantarell-Regular.ttf']
        found = None
        for root in ('/usr/share/fonts', '/run/host/usr/share/fonts', str(Path.home() / '.local/share/fonts')):
            for c in cands:
                hit = next((q for q in Path(root).rglob(c)), None) if Path(root).is_dir() else None
                if hit: found = hit; break
            if found: break
        if not found:
            found = next((q for root in ('/usr/share/fonts', '/run/host/usr/share/fonts') if Path(root).is_dir() for q in Path(root).rglob('*.ttf')), None)
        if found: shutil.copyfile(found, font / 'standard.ttf'); say(f'standard font: {found}')
        else: say('[RED] no TTF font found (sudo dnf install -y google-noto-sans-fonts): the game will have no text')
    if not (font / 'nintendo_ext.ttf').exists():
        try:
            import fontTools  # noqa: F401
            run([sys.executable, REPO / 'dm/tools/make_ext_font.py', font / 'nintendo_ext.ttf'] + ([font / 'standard.ttf'] if (font / 'standard.ttf').exists() else []))
        except ImportError: say(f'[YELLOW] no button-icon font: {sys.executable} -m pip install --user fonttools, then build again')
    if a.shaders or a.retranslate: shaders(tid, ver, ext, exefs, nro_dir, lw, renv, a.jobs, a.retranslate)
    aoc = work / 'aoc'
    script = lw / 'run.sh'
    script.write_text(f'''#!/bin/sh
# Runs {tid} v{ver} with the null graphics device (headless). The log goes to the screen and to log.txt beside this script.
# Ctrl+C stops it. DM_TIME_LIMIT=<seconds> stops it by itself.
cd "$(dirname "$0")"
export DM_ROOT="{run_dir}" DM_ROMFS="{ext / 'RomFS.img'}" DM_SAVE="{run_dir}/save" DM_TEMP="{run_dir}/temp" DM_FONT_DIR="{lw}/font"
export DM_AUDIO="${{DM_AUDIO:-1}}"   # sound through SDL3 (DM_AUDIO=0: silent)
export DM_OPMODE="${{DM_OPMODE:-0}}"   # handheld mode: the game renders for a handheld screen (Switch 2 docked mode is 1440p: four times the render targets' memory and GPU time on a 1080p screen); DM_OPMODE=1 docked
{f'export DM_SHADERS="{lw / "shaders.dmsp"}"   # the translated shaders (stage C)' if (lw / 'shaders.dmsp').is_file() else '# no shader pack yet (build with --shaders)'}
{f'export DM_AOC_DIR="{aoc}"' if aoc.is_dir() and any(aoc.iterdir()) else '# no DLC'}
exec ./dmhost "{patched}" "$@" 2>&1 | tee log.txt
''')
    script.chmod(0o755)
    # play.sh: the launcher to add to Steam (Add a Non-Steam Game). Steam runs on the host, outside the toolbox the game was built in: the launcher enters the toolbox
    # (its libraries: SDL3, Vulkan, .NET are not needed to play) and starts the game full screen with its window. Under Steam the controls come through Steam Input
    # as an Xbox 360 pad, which also works where the handheld's own controls act as a mouse (desktop mode)
    box = ''
    try:
        for line in open('/run/.containerenv'):
            if line.startswith('name='): box = line.split('=', 1)[1].strip().strip('"')
    except OSError: pass
    play = lw / 'play.sh'
    play.write_text(f'''#!/bin/bash
# Starts {tid} full screen with controllers. Add this file to Steam: Games > Add a Non-Steam Game to My Library > Browse > this file.
# Runs inside the toolbox "{box or 'none'}" (where it was built). What the launcher did goes to play.log beside this file, the game's log to log.txt.
# DM_PAD_LAYOUT=xbox in Steam's launch options (DM_PAD_LAYOUT=xbox %command%): A at the bottom
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec >>"$here/play.log" 2>&1
echo "=== $(date) play.sh in: $( [ -f /run/.containerenv ] && echo container || echo host ) ($( [ -f /.flatpak-info ] && echo flatpak ))"
if [ -f /.flatpak-info ]; then   # Steam from Flathub: out of its sandbox first (needs: flatpak override --user --talk-name=org.freedesktop.Flatpak com.valvesoftware.Steam)
    exec flatpak-spawn --host "$here/play.sh" "$@"
fi
# Steam gives a game its controller layout only while the window in front belongs to a process it started: the game runs straight on the host when the host has
# what it loads (SDL3, the Vulkan loader; the GPU driver is the system's), and goes through the toolbox only when not (DM_TOOLBOX=1 forces it)
# The CPU at full speed while the game runs (the Thor's automatic setting cost about a fifth of the frame rate), back to what it was when the game ends: only when
# scripts/linux_cpu_setup.sh was run once (it allows this one helper without a password)
cpu_on=0
if [ ! -f /run/.containerenv ] && [ -x /etc/nxnative/nxnative-cpu ] && [ "${{DM_CPU_FULL:-1}}" != 0 ] && ! grep -qs "^DM_CPU_FULL=0" "$HOME/.config/nxnative/env" "$HOME/.config/nxnative/{tid}.env" && sudo -n /etc/nxnative/nxnative-cpu on 2>/dev/null; then
    cpu_on=1; echo "CPU at full speed for the game (DM_CPU_FULL=0 leaves it alone)"
fi
cpu_off() {{ if [ "$cpu_on" = 1 ]; then sudo -n /etc/nxnative/nxnative-cpu off 2>/dev/null; cpu_on=0; echo "CPU speed setting restored"; fi; }}
trap 'cpu_off' EXIT
trap 'cpu_off; exit 143' TERM
trap 'cpu_off; exit 130' INT
trap 'cpu_off; exit 129' HUP
if [ ! -f /run/.containerenv ] && [ -n "{box}" ]; then
    need_box="${{DM_TOOLBOX:-0}}"
    if [ "$need_box" != 1 ]; then
        for lib in libSDL3.so.0 libvulkan.so.1; do
            if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then echo "the host has no $lib: through the toolbox"; need_box=1; fi
        done
        if [ "$need_box" != 1 ] && ldd "$here/dmhost" 2>&1 | grep -q "not found\\|version .* not found"; then echo "dmhost does not load on the host: through the toolbox"; ldd "$here/dmhost"; need_box=1; fi
    fi
    if [ "$need_box" = 1 ]; then
        unset LD_PRELOAD LD_LIBRARY_PATH   # Steam's own environment does not belong in the toolbox
        tb="$(command -v toolbox || echo /usr/bin/toolbox)"
        echo "entering toolbox {box} with $tb"
        "$tb" run -c "{box}" "$here/play.sh" "$@"; exit $?
    fi
    echo "running on the host"
fi
cd "$here"
# Settings to try without Steam's launch options: one NAME=value per line in ~/.config/nxnative/env (every game), then ~/.config/nxnative/{tid}.env (this game only,
# its lines win); lines starting with # are ignored
for ef in "$HOME/.config/nxnative/env" "$HOME/.config/nxnative/{tid}.env"; do
    [ -f "$ef" ] || continue
    while IFS= read -r line; do
        case "$line" in ''|'#'*) ;; *=*) export "$line"; echo "setting from $ef: $line" ;; esac
    done < "$ef"
done
export DM_WINDOW=1 DM_FULLSCREEN="${{DM_FULLSCREEN:-1}}"
# Steam's Fossilize layer (present when Steam or Gamescope starts the game) records every shader module and pipeline: with the game's multi-megabyte shaders it held gigabytes the AYN Thor
# does not have. DM_FOSSILIZE=1 keeps it
[ "${{DM_FOSSILIZE:-0}}" = 1 ] || export DISABLE_VK_LAYER_VALVE_steam_fossilize_1=1
echo "starting the game (DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR)"
# A memory guard: the game runs in a systemd scope capped at 75 % of the RAM and 3 GB of swap, so when it outgrows the machine the kernel stops the game, not the whole system
# (Ocarina of Time's first scene froze the AYN Thor until it was restarted). DM_MEM_GUARD=0 runs without it; DM_MEM_MAX / DM_SWAP_MAX set the caps (systemd sizes, e.g. 8G)
runner="./run.sh"
if [ "${{DM_MEM_GUARD:-1}}" = 1 ] && command -v systemd-run >/dev/null 2>&1; then
    tot=$(awk '/MemTotal/ {{print int($2 / 1024)}}' /proc/meminfo)
    mmax="${{DM_MEM_MAX:-$((tot * 3 / 4))M}}"; smax="${{DM_SWAP_MAX:-3G}}"
    if systemd-run --user --scope --quiet -p MemoryMax=1G true >/dev/null 2>&1; then
        echo "memory guard: the game may use $mmax of RAM and $smax of swap (DM_MEM_GUARD=0 turns it off)"
        runner="systemd-run --user --scope --quiet -p MemoryMax=$mmax -p MemorySwapMax=$smax ./run.sh"
    else echo "memory guard: systemd-run --user does not work here; running without it"; fi
fi
if [ "$cpu_on" = 1 ]; then $runner "$@"; st=$?; [ $st = 137 ] && echo "the game was stopped by the memory guard (out of memory)"; exit $st; fi
exec $runner "$@"
''')
    play.chmod(0o755)
    say(f'\ndone. Run it with:\n  {script}\nthen send me {lw / "log.txt"}')
    say(f'To play from Steam (controllers through Steam Input): add this as a non-Steam game:\n  {play}')


if __name__ == '__main__':
    main()
