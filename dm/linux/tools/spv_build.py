#!/usr/bin/env python3
# The game's shaders as SPIR-V, on Linux: dm/gfx/tools/dmsh_build.py run unchanged (its job preparation: the control sections, the facts table, vertex A/B merging, transform
# feedback layouts…), with the translator (dmsh_xlate, Ryujinx's, .NET 10) through `dotnet`, and the Mac-only steps after it (spirv-cross → MSL → metallib) failing at once.
# The translator's SPIR-V and its job (stage and the facts it was given) are caught as they are written: <out>/<key>.spv and <key>.json, what the Vulkan layer (stage C) loads.
# The chain files stay as they are, so the chain version and dm/gfx/nvn/dmsh_chain.h match upstream nxnative's.
#
# Usage: spv_build.py <shader package dir (shader_split.py's)> <out dir> [key...]   (DM_GAME = the profile, as for dmsh_build.py)
# Needs: the .NET 10 runtime (Fedora: sudo dnf install -y dotnet-runtime-10.0)
import json, os, runpy, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / 'dm/gfx/tools'

if len(sys.argv) >= 2 and sys.argv[1] == '--xlate':
    # The translator, as dmsh_build.py calls it (XLATE <jobs.json> <work dir>), then a copy of what it wrote
    jp, w, out = sys.argv[2], sys.argv[3], os.environ['DM_SPV_OUT']
    r = subprocess.run(['dotnet', str(TOOLS / 'dmsh_xlate_bin/dmsh_xlate.dll'), jp, w], capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    try: jobs = json.load(open(jp))
    except (OSError, ValueError): jobs = []
    for j in jobs if isinstance(jobs, list) else [jobs]:
        k = j.get('key'); spv = os.path.join(w, f'{k}.spv')
        if not k or not os.path.exists(spv): continue
        shutil.copyfile(spv, os.path.join(out, f'{k}.spv'))
        res = os.path.join(w, f'{k}.res.json')   # the translator's reflection: each buffer's, texture's and image's binding (all in set 0)
        if os.path.exists(res): shutil.copyfile(res, os.path.join(out, f'{k}.res.json'))
        j = dict(j); j.pop('code', None)   # the code file is the work folder's
        with open(os.path.join(out, f'{k}.json'), 'w') as f: json.dump(j, f)
    sys.exit(r.returncode)

if len(sys.argv) < 3: sys.exit('usage: spv_build.py <shader package dir> <out dir> [key...]   (DM_GAME=<profile>)')
if not os.environ.get('DM_GAME'): sys.exit('[RED] DM_GAME is not set (the game profile, e.g. 0400D4201C6EA000)')
if not shutil.which('dotnet'): sys.exit('[RED] no .NET runtime: sudo dnf install -y dotnet-runtime-10.0 (in the toolbox)')
pkg, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
v = subprocess.run(['dotnet', str(TOOLS / 'dmsh_xlate_bin/dmsh_xlate.dll'), '--source-sha'], capture_output=True, text=True)
if v.returncode: sys.exit(f'[RED] the translator does not start: {(v.stderr or v.stdout).strip()[:300]}')
print(f'translator {v.stdout.strip()} on .NET ({shutil.which("dotnet")})', flush=True)

# Stand-ins for the Mac tools: the Metal compiler and spirv-cross fail (dmsh_build.py then records those shaders as failed after their SPIR-V was caught)
fake = Path(tempfile.mkdtemp(prefix='spv_build_'))
# dmsh_build.py asks xcrun where the Metal tools are when it starts (-f metal, --show-sdk-path): answered with stand-ins that fail when run
(fake / 'xcrun').write_text('#!/bin/sh\nfor a; do case "$a" in --show-sdk-path) echo /nonexistent-macos-sdk; exit 0;; -f) echo "$(dirname "$0")/no-metal"; exit 0;; esac; done\n'
                            'echo "no Metal compiler on Linux (SPIR-V capture)" >&2\nexit 1\n')
(fake / 'no-metal').write_text('#!/bin/sh\necho "no Metal compiler on Linux (SPIR-V capture)" >&2\nexit 1\n')
(fake / 'spirv-cross').write_text('#!/bin/sh\n[ "$1" = "--version" ] && { echo "spirv-cross (none: Linux SPIR-V capture)"; exit 0; }\nexit 1\n')
(fake / 'xlate').write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" --xlate "$@"\n')
for f in fake.iterdir(): f.chmod(0o755)
os.environ['PATH'] = f'{fake}{os.pathsep}{os.environ.get("PATH", "")}'
os.environ['SPIRV_CROSS'] = str(fake / 'spirv-cross')
os.environ['DM_SPV_OUT'] = str(out)
os.environ.setdefault('DMSH_CAS', str(out / '.mac_cache.db'))   # dmsh_build.py's compile cache: nothing Mac-side lands in it, but it must not be shared with a Mac build's

sys.path.insert(0, str(TOOLS))
import dmsh_chain, tool_lock, dmsh_mtlcache   # noqa: E402  dmsh_chain first: its XLATE_DIR stays the in-repo translator's
tool_lock.check_tools = lambda: []            # the lock records the Mac's spirv-cross and Metal compilers; the translator was checked above (--source-sha)
dmsh_mtlcache.compiler_id = lambda sdk: 'linux-none'
os.environ['XLATE'] = str(fake / 'xlate')     # read by dmsh_build.py at its start
sys.argv = ['dmsh_build.py', str(pkg), str(out / '.mac_out')] + sys.argv[3:]
try:
    runpy.run_path(str(TOOLS / 'dmsh_build.py'), run_name='__main__')
except SystemExit as e:
    if e.code not in (0, None, 1): print(f'dmsh_build.py ended with {e.code}', flush=True)
finally:
    shutil.rmtree(fake, ignore_errors=True)
    shutil.rmtree(out / '.mac_out', ignore_errors=True)
n = sum(1 for _ in out.glob('*.spv')); nr = sum(1 for _ in out.glob('*.res.json'))
print(f'\nSPIR-V: {n} shaders ({nr} with bindings) in {out}', flush=True)
sys.exit(0 if n else 1)
