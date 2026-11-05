#!/usr/bin/env python3
# Unpack a Switch 2 NUP container (one .nup file, https://github.com/a55uka/nup SPEC v1) into a plain folder of loose files,
# in the dump layout build_ipa_from_nsp.py / pack_dump.py read:
#
#   <out>/base/exefs/                  main, main.npdm, sdk, rtld, ...
#   <out>/base/romfs/                  the game's data files
#   <out>/base/control.nacp, icon.*    when the NUP carries them
#   <out>/updates/<version>/exefs/     each update (decimal version), its RomFS as stored (the patched RomFS)
#   <out>/updates/<version>/romfs/
#   <out>/dlc/<dlc title ID>/0/romfs/  each DLC
#
# nxnative itself can take the .nup directly; this is for looking inside a game, or for tools that want loose files (Lichii, Ghidra, ...).
# NUP files are not encrypted, so no keys are needed. Every file is checked against its SHA-256 while it is written.
# Needs Python 3.14+ (compression.zstd) or `pip install backports.zstd`.
#
# Usage:
#   python3 scripts/nup_extract.py Game.nup                 → ./Game/ next to the .nup
#   python3 scripts/nup_extract.py Game.nup -o ~/Games/OoT
#   python3 scripts/nup_extract.py Game.nup --list          just show what is inside
#   python3 scripts/nup_extract.py Game.nup --exefs-only    only the code (small: what a recomp needs to start)
import argparse, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_nup
from pack_nup import Error, NO_DIR, BASE, UPDATE, ADDON, T_NACP, T_ICON


def safe(name, where):   # a NUP name as one path component, refusing anything that could escape the output folder
    s = os.fsdecode(name)
    if s in ('', '.', '..') or '\0' in s or '/' in s or '\\' in s:
        raise Error(f'{where}: unsafe name {name!r} in the container')
    return s


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB': return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1024


class Progress:
    def __init__(self, total): self.total, self.done, self.last = max(total, 1), 0, -1
    def add(self, n):
        self.done += n
        pct = self.done * 100 // self.total
        if pct != self.last:
            self.last = pct
            print(f'\r  {pct:3d}%  {human(self.done)} of {human(self.total)}', end='', flush=True)
    def end(self): print()


def unpack_tree(n, d, dst, prog, where):   # container folder d → dst on disk, recursively
    dst.mkdir(parents=True, exist_ok=True)
    for name, isdir, i in n.children(d):
        p = dst / safe(name, where)
        if isdir:
            unpack_tree(n, i, p, prog, where)
            continue
        if p.exists() and p.stat().st_size == n.files[i][1]:   # resume: keep a file already written whole (its hash was checked then)
            prog.add(n.files[i][1]); continue
        tmp = p.with_name(p.name + '.part')
        with open(tmp, 'wb') as f: n.copy(i, f)
        tmp.replace(p)
        prog.add(n.files[i][1])


def write_control(k, dst):
    m = k.meta()
    if T_NACP in m: (dst / 'control.nacp').write_bytes(m[T_NACP])
    if T_ICON in m:
        icon = m[T_ICON]
        ext = 'png' if icon[:4] == b'\x89PNG' else 'jpg'
        (dst / f'icon.{ext}').write_bytes(icon)


def where_to(k, out):
    if k.kind == BASE: return out / 'base'
    if k.kind == UPDATE: return out / 'updates' / str(k.version)
    if k.kind == ADDON: return out / 'dlc' / f'{k.title_id:016x}' / '0'
    return None   # addon updates and system packages: not part of the dump layout


def main():
    ap = argparse.ArgumentParser(description='Unpack a Switch 2 .nup file into a folder of loose files (base/, updates/, dlc/)')
    ap.add_argument('nup', type=Path)
    ap.add_argument('-o', '--out', type=Path, help='output folder (default: next to the .nup, named after it)')
    ap.add_argument('--list', action='store_true', help='only list the packages inside')
    ap.add_argument('--exefs-only', action='store_true', help='only the code modules (exefs), no game data')
    a = ap.parse_args()

    if not pack_nup.is_nup(a.nup):
        sys.exit(f'{a.nup}: not a .nup file')
    with pack_nup.Nup(a.nup) as n:
        print(f'{a.nup.name}: {len(n.packages)} package(s)')
        for k in n.packages:
            size = n.tree_bytes(k.root)
            print(f'  {pack_nup.KINDS[k.kind]:<12} title {k.title_id:016X}  version {k.version:<8} {human(size)}')
        if a.list: return

        out = a.out or a.nup.with_suffix('')
        jobs = []   # (package, folder)
        for k in n.packages:
            dst = where_to(k, out)
            if dst is None:
                print(f'  skipping {k.name}: not part of the dump layout'); continue
            if k.kind == BASE and not n.tree_bytes(k.root):
                continue   # an empty base record (the update is the whole game)
            jobs.append((k, dst))
        if not jobs: sys.exit('nothing to unpack')

        def part_size(k):
            s = 0
            if k.exefs != NO_DIR: s += n.tree_bytes(k.exefs)
            if k.romfs != NO_DIR and not a.exefs_only: s += n.tree_bytes(k.romfs)
            return s
        total = sum(part_size(k) for k, _ in jobs)
        free = shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free
        if total > free:
            sys.exit(f'not enough free space: need {human(total)}, have {human(free)}')

        print(f'unpacking into {out}/')
        prog = Progress(total)
        for k, dst in jobs:
            dst.mkdir(parents=True, exist_ok=True)
            if k.exefs != NO_DIR: unpack_tree(n, k.exefs, dst / 'exefs', prog, k.name)
            if k.romfs != NO_DIR and not a.exefs_only: unpack_tree(n, k.romfs, dst / 'romfs', prog, k.name)
            write_control(k, dst)
        prog.end()
        print('done.')
        base = out / 'base' / 'exefs'
        if base.is_dir():
            print(f'code modules: {", ".join(sorted(p.name for p in base.iterdir()))}')


if __name__ == '__main__':
    try: main()
    except Error as e: sys.exit(f'error: {e}')
    except KeyboardInterrupt: sys.exit('\nstopped (run again to resume)')
