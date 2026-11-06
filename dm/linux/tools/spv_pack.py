#!/usr/bin/env python3
# spv_build.py's output (<key>.spv + <key>.res.json per shader, over 100 000 small files) → one file the Linux host maps (dm/linux/gfx_vk_shader.c), shaders.dmsp:
#   header  "DMSP" u32 version (2), u32 count, u32 0
#   index   count × {u64 key (the 16 hex digits read as a big-endian number), u32 offset, u32 length}, sorted by key
#   entries u8 stage (SPIR-V execution model: 0 vertex 1 tess control 2 tess eval 3 geometry 4 fragment 5 compute), u8 bindings, u8 vertex inputs, u8 fragment output mask,
#           u16 local size x, y, z (compute), u16 c1 bytes
#           bindings × {u8 binding, u8 Vulkan descriptor type, u8 kind (KIND_*), u8 image (dim | arrayed << 4 | multisample << 5 | depth << 6 | integer texels << 7), u32 reg, u16 count, u16 aux}
#             KIND_CBUF: reg = the constant bank (0, 2 driver; 1 the shader's own c1; 3 + n uniform slot n)
#             KIND_SBUF: reg = the byte its address is at in bank aux (aux 0: the stage's c0 table; aux >= 3: uniform slot aux - 3); image = 1 when the shader writes it
#             KIND_TEX / KIND_IMG: reg = the translator's handle (bits 0-13 the handle word, 14-27 a separate sampler's word, 28-31 the handle type), aux = the bank the words are in
#           vertex inputs × {u8 location, u8 component type (0 float 1 uint 2 int)}, the c1 bytes, padded to 4 bytes, then the SPIR-V words
# The descriptor types and image shapes come from the SPIR-V itself; what each binding means (which uniform slot, which texture handle) from the translator's reflection.
# All bindings are in set 0 (dmsh_xlate Program.cs); binding 0 is the translator's support buffer.
import json, os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spv_passthrough
from pathlib import Path

KIND_SUPPORT, KIND_CBUF, KIND_SBUF, KIND_TEX, KIND_IMG, KIND_UNKNOWN = range(6)

def reflect(words):
    names = {}; dec = {}; types = {}; variables = []; entry = None; local = (0, 0, 0)
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        o = words[i + 1:i + n]
        if n == 0: raise ValueError('bad SPIR-V')
        if op == 15: entry = o[0]                                        # OpEntryPoint
        elif op == 16 and o[1] == 17: local = tuple(o[2:5])             # OpExecutionMode LocalSize
        elif op == 71: dec.setdefault(o[0], {})[o[1]] = o[2] if len(o) > 2 else 1   # OpDecorate
        elif op in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32): types[o[0]] = (op, o[1:])
        elif op == 59: variables.append((o[0], o[1], o[2]))             # OpVariable type id storage
        i += n
    def base(t):
        while True:
            op, a = types[t]
            if op == 32: t = a[1]                                       # pointer → pointee
            elif op in (28, 29): t = a[0]                               # (runtime) array → element
            else: return t
    def count(t):
        op, a = types[t]
        if op == 32: return count(a[1])
        return 1 if op != 28 else 0                                     # fixed arrays: filled in by the caller
    binds, vin, fout = [], [], 0
    for tid, vid, sc in variables:
        d = dec.get(vid, {})
        bt = base(tid); op, a = types[bt]
        if sc in (0, 2, 12) and 33 in d:                                # UniformConstant / Uniform / StorageBuffer with a Binding
            img = 0
            if op == 27: op, a = types[a[0]]; vk = 1                    # sampled image: combined image sampler
            if op == 25:                                                # OpTypeImage: sampled type, dim, depth, arrayed, ms, sampled, format
                dim, depth, arr, ms, sampled = a[1], a[2], a[3], a[4], a[5]
                integer = types.get(a[0], (0,))[0] == 21                 # its texels are read as integers (the host's view format follows: gfx_vk_draw.inc tex_view)
                img = (dim & 15) | arr << 4 | ms << 5 | (depth == 1) << 6 | integer << 7
                if types[bt][0] != 27: vk = (4 if sampled == 1 else 5) if dim == 5 else (2 if sampled == 1 else 3)
                elif dim == 5: vk = 4
            elif op == 26: vk = 0                                       # sampler
            elif op == 30:
                vk = 7 if sc == 12 or 3 in dec.get(bt, {}) else 6      # StorageBuffer storage, or BufferBlock (decoration 3) → storage; Block → uniform
            else: raise ValueError(f'binding {d[33]}: type op {op}')
            pt = types[tid][1][1]; n = 1
            if types[pt][0] == 28:                                       # array of descriptors: its length is a constant; read below
                n = ('const', types[pt][1][1])
            binds.append([d[33], vk, img, n])
        elif sc == 1 and 30 in d and 11 not in d:                       # Input with a Location, not a built-in
            t = base(tid); top, ta = types[t]
            if top == 23: t = ta[0]; top, ta = types[t]                 # vector → component
            vin.append((d[30], 1 if top == 21 and ta[1] == 0 else 2 if top == 21 else 0))
        elif sc == 3 and 30 in d and 11 not in d: fout |= 1 << d[30]
    consts = {}
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if op == 43: consts[words[i + 2]] = words[i + 3]               # OpConstant
        i += n
    for b in binds:
        if isinstance(b[3], tuple): b[3] = consts.get(b[3][1], 1)
    model = words[[j for j in range(5, len(words)) if words[j] & 0xffff == 15][0] + 1]
    return model, binds, vin, fout, local

def kind_of(res, binding):
    if binding == 0: return KIND_SUPPORT, 0, 0, 0
    for b in res.get('cbufs', []):
        if b['binding'] == binding: return KIND_CBUF, b['slot'], 0, 0
    for b in res.get('sbufs', []):
        if b['binding'] == binding: return KIND_SBUF, b['cb_offset'] * 4, b['cb_slot'] & 0xffff, b.get('flags', 0) & 1
    for name, kind in (('textures', KIND_TEX), ('images', KIND_IMG)):
        for b in res.get(name, []):
            if b['binding'] == binding: return kind, b['handle'] & 0xffffffff, b['cbuf_slot'] & 0xffff, 0
    return KIND_UNKNOWN, 0, 0, 0

def add_image_caps(words):
    # Storage images without a format (the translator's, OpTypeImage ... Unknown) read or written need StorageImageReadWithoutFormat / WriteWithoutFormat, which the
    # translator does not declare (Metal does not ask): a driver may read zeros without them (Turnip). Declared after the other capabilities when such an image is used
    unknown_storage, have = False, set()
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if op == 17: have.add(words[i + 1])
        if op == 25 and words[i + 7] == 2 and words[i + 8] == 0: unknown_storage = True   # OpTypeImage, Sampled 2 (storage), Format Unknown
        i += n
    if not unknown_storage: return words
    add = [c for c in (55, 56) if c not in have]   # StorageImageReadWithoutFormat, StorageImageWriteWithoutFormat
    if not add: return words
    out = list(words[:5]); i = 5; done = False
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if op != 17 and not done:
            for c in add: out += [2 << 16 | 17, c]
            done = True
        out += words[i:i + n]; i += n
    return out

def strip_unused(words):
    # The translator declares Float64 and fragment shader interlock in every shader; Adreno's driver (Turnip) supports neither. Declarations a shader does not use are dropped
    used64 = interlock = False
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if op == 22 and words[i + 2] == 64: used64 = True                                  # OpTypeFloat 64
        if op in (5364, 5365) or (op == 16 and 5366 <= words[i + 2] <= 5371): interlock = True   # Begin / EndInvocationInterlockEXT, the interlock execution modes
        i += n
    out = list(words[:5]); i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        ins = words[i:i + n]
        drop = (op == 17 and ins[1] == 10 and not used64) or (op == 17 and ins[1] in (5378, 5379, 5380) and not interlock) or \
               (op == 10 and not interlock and struct.pack(f'<{n - 1}I', *ins[1:]).rstrip(b'\0') == b'SPV_EXT_fragment_shader_interlock')
        if not drop: out += ins
        i += n
    return out

def entry(spv_path, res_path, job_path=None):
    data = Path(spv_path).read_bytes()
    words = struct.unpack(f'<{len(data) // 4}I', data)
    if words[0] != 0x07230203: raise ValueError('not SPIR-V')
    if spv_passthrough.is_passthrough(words):   # NVIDIA passthrough geometry shader → an ordinary one (spv_passthrough.py); one it cannot rewrite stays as it is
        try: words = spv_passthrough.rewrite(words)
        except Exception as e: print(f'[YELLOW] {Path(spv_path).stem}: passthrough geometry shader not rewritten ({e}): kept as it is', flush=True)
    words = add_image_caps(strip_unused(words)); data = struct.pack(f'<{len(words)}I', *words)
    model, binds, vin, fout, local = reflect(words)
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    try: c1 = bytes.fromhex(json.load(open(job_path)).get('c1') or '') if job_path and os.path.exists(job_path) else b''
    except ValueError: c1 = b''
    out = bytearray(struct.pack('<BBBB3HH', model, len(binds), len(vin), fout & 0xff, *[min(x, 0xffff) for x in local], len(c1)))
    for b, vk, img, n in sorted(binds):
        k, reg, aux, written = kind_of(res, b)
        if k == KIND_SBUF: img = written                                # buffers have no image shape: the byte says whether the shader writes it
        out += struct.pack('<BBBBIHH', b, vk, k, img, reg, n, aux)
    for loc, ct in sorted(vin): out += struct.pack('<BB', loc, ct)
    out += c1
    out += b'\0' * (-len(out) % 4)
    return bytes(out + data)

def main():
    if len(sys.argv) != 3: sys.exit('usage: spv_pack.py <spv dir> <out file>')
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    keys = sorted(p.name[:-4] for p in os.scandir(src) if p.name.endswith('.spv') and len(p.name) >= 20)
    # <key>_g0 .. _g4 are the Mac's geometry lowering (geometry shaders made compute kernels): not packed
    blobs, bad = [], 0
    for k in keys:
        try: key = int(k[:16], 16)
        except ValueError: continue
        if k != k[:16]: continue
        try: blobs.append((key, entry(src / f'{k}.spv', src / f'{k}.res.json', src / f'{k}.json')))
        except Exception as e:
            bad += 1
            if bad <= 10: print(f'[RED] {k}: {e}')
        if len(blobs) % 5000 == 0 and blobs: print(f'packed {len(blobs)}/{len(keys)}', flush=True)
    off = 16 + 16 * len(blobs)
    idx, body = bytearray(), bytearray()
    blobs.sort(key=lambda kb: kb[0])
    for key, b in blobs:
        idx += struct.pack('<QII', key, off + len(body), len(b)); body += b; body += b'\0' * (-len(body) % 8)
    tmp = dst.with_suffix('.partial')
    with open(tmp, 'wb') as f: f.write(b'DMSP' + struct.pack('<III', 2, len(blobs), 0)); f.write(idx); f.write(body)
    tmp.replace(dst)
    print(f'shader pack: {len(blobs)} shaders, {bad} unreadable, {(16 + len(idx) + len(body)) / 2**20:.1f} MB → {dst}')

if __name__ == '__main__': main()
