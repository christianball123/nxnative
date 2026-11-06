# NVN's alpha test (nvnColorStateSetAlphaTest + nvnCommandBufferSetAlphaRef: Maxwell's fixed-function test of the alpha a pixel shader writes to colour 0, before
# blending) has no Vulkan state. Every fragment shader that writes a float vec4 to location 0 and has the translator's support buffer gets the test as code: two
# members appended to the support buffer block (uint function at byte 1392, float reference at 1396, filled per draw by gfx_vk_draw.inc support_buffer), and
# before each return of the entry function colour 0's alpha is compared and the fragment discarded when the test fails. Function 0 (no test) and 8 (ALWAYS)
# pass, as the NVN values: NEVER 1, LESS 2, EQUAL 3, LEQUAL 4, GREATER 5, NOTEQUAL 6, GEQUAL 7, ALWAYS 8.
# Mario Tennis Fever's effects (shot trails, hit flashes) draw with it: without it their quads showed as solid blocks.
import struct

AT_FUNC_OFF, AT_REF_OFF = 1392, 1396
AT_SPEC_ID = 200   # a bool specialization constant: true in pipelines drawn with an alpha test (gfx_vk_draw.inc pipe_make)
OP_TYPES = set(range(19, 40)) | {322, 4456, 5341}   # type declarations (start of the types section)
OP_ANNOT = {71, 72, 73, 74, 75, 332, 5632, 5633}

def _ins(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0: raise ValueError('bad SPIR-V')
        yield i, words[i] & 0xffff, words[i:i + n]
        i += n

def add_alpha_test(words):
    """Returns (words, done): done is False (words unchanged) for shaders the test does not apply to."""
    words = list(words)
    entry = None; types = {}; ptrs = {}; vars_ = {}; decos = {}; mdecos = {}
    for i, op, ins in _ins(words):
        if op == 15 and ins[1] == 4: entry = ins[2]            # OpEntryPoint Fragment
        elif op in (19, 20, 21, 22, 23, 24, 28, 29, 30): types[ins[1]] = (op, list(ins[2:]))
        elif op == 32: ptrs[ins[1]] = (ins[2], ins[3])
        elif op == 59: vars_[ins[2]] = (ins[1], ins[3])
        elif op == 71: decos.setdefault(ins[1], []).append(list(ins[2:]))
        elif op == 72: mdecos.setdefault(ins[1], []).append(list(ins[2:]))
    if entry is None: return words, False
    def deco(v, d): return next((x[1:] for x in decos.get(v, []) if x[0] == d), None)
    def find_type(op, ops):
        return next((t for t, (o, a) in types.items() if o == op and a == ops), None)
    f32 = find_type(22, [32]); u32 = find_type(21, [32, 0]); b = find_type(20, [])
    if f32 is None: return words, False
    v4 = find_type(23, [f32, 4])
    # colour 0: an Output vec4 float variable at Location 0 (Index 0, or none)
    out = None
    for v, (pt, sc) in vars_.items():
        if sc != 3 or deco(v, 30) != [0] or (deco(v, 32) or [0]) != [0]: continue
        if pt in ptrs and ptrs[pt][1] == v4 and v4 is not None: out = v; break
    if out is None: return words, False
    # the support buffer: the Uniform block at set 0 binding 0
    sb = st = None
    for v, (pt, sc) in vars_.items():
        if sc == 2 and deco(v, 33) == [0] and deco(v, 34) == [0] and pt in ptrs:
            s = ptrs[pt][1]
            if s in types and types[s][0] == 30: sb, st = v, s; break
    if sb is None: return words, False
    nmem = len(types[st][1])
    if any(m[0] >= nmem for m in mdecos.get(st, [])) or nmem != 8: return words, False   # not the layout this was written for
    offs = {m[0]: m[2] for m in mdecos.get(st, []) if m[1] == 35}
    if offs.get(nmem - 1, 1 << 30) >= AT_FUNC_OFF: return words, False
    bound = words[3]
    def nid():
        nonlocal bound
        bound += 1
        return bound - 1
    pre_types = []   # scalar types the shader lacks, before every other type
    if u32 is None: u32 = nid(); pre_types += [3 << 16 | 21, u32, 32, 0]
    if b is None: b = nid(); pre_types += [2 << 16 | 20, b]
    # globals appended before the first function
    glob = []
    def ptr(sc, t):
        i = nid(); glob.extend([4 << 16 | 32, i, sc, t]); return i
    def const(t, v):
        i = nid(); glob.extend([4 << 16 | 43, t, i, v]); return i
    p_out_f, p_uni_u, p_uni_f = ptr(3, f32), ptr(2, u32), ptr(2, f32)
    c3, c8, c9 = const(u32, 3), const(u32, nmem), const(u32, nmem + 1)
    cf = {k: const(u32, k) for k in range(0, 9)}
    en = nid(); glob.extend([3 << 16 | 49, b, en])   # OpSpecConstantFalse, SpecId AT_SPEC_ID: the pipeline turns the test on (off, the driver drops it and the shader keeps early depth)
    annot = [4 << 16 | 71, en, 1, AT_SPEC_ID, 5 << 16 | 72, st, nmem, 35, AT_FUNC_OFF, 5 << 16 | 72, st, nmem + 1, 35, AT_REF_OFF]
    # rebuild
    out_w = list(words[:5]); in_entry = False; n_ret = 0; placed_annot = placed_types = placed_glob = False
    for i, op, ins in _ins(words):
        if not placed_annot and (op in OP_TYPES or op in (43, 59, 54) ) :
            out_w += annot; placed_annot = True
        if not placed_types and op in OP_TYPES:
            out_w += pre_types; placed_types = True
        if op == 30 and ins[1] == st:   # the support block: two members more
            ins = [(len(ins) + 2) << 16 | 30] + list(ins[1:]) + [u32, f32]
        if op == 54 and not placed_glob:   # OpFunction
            out_w += glob; placed_glob = True
        if op == 54: in_entry = ins[2] == entry
        elif op == 56: in_entry = False
        if op == 253 and in_entry:   # OpReturn in the entry function: the test first
            pa, a, pf, f, pr, r = nid(), nid(), nid(), nid(), nid(), nid()
            out_w += [5 << 16 | 65, p_out_f, pa, out, c3, 4 << 16 | 61, f32, a, pa,
                      5 << 16 | 65, p_uni_u, pf, sb, c8, 4 << 16 | 61, u32, f, pf,
                      5 << 16 | 65, p_uni_f, pr, sb, c9, 4 << 16 | 61, f32, r, pr]
            def nid_not(x):
                i2 = nid(); out_w.extend([4 << 16 | 168, b, i2, x]); return i2
            def bop(o, x, y):
                i2 = nid(); out_w.extend([5 << 16 | o, b, i2, x, y]); return i2
            pas = bop(170, f, cf[0]); pas = bop(166, pas, bop(170, f, cf[8]))
            for fn, cmp in ((2, 184), (3, 180), (4, 188), (5, 186), (6, 182), (7, 190)):
                pas = bop(166, pas, bop(167, bop(170, f, cf[fn]), bop(cmp, a, r)))
            pas = bop(166, pas, nid_not(en))
            m, k = nid(), nid()
            out_w += [3 << 16 | 247, m, 0, 4 << 16 | 250, pas, m, k, 2 << 16 | 248, k, 1 << 16 | 252, 2 << 16 | 248, m]
            n_ret += 1
        out_w += list(ins)
    if not n_ret or not placed_glob: return words, False
    out_w[3] = bound
    return out_w, True

if __name__ == '__main__':
    import sys
    w = list(struct.unpack('<%dI' % (len(open(sys.argv[1], 'rb').read()) // 4), open(sys.argv[1], 'rb').read()))
    o, ok = add_alpha_test(w)
    print('done' if ok else 'not applied')
    if ok and len(sys.argv) > 2: open(sys.argv[2], 'wb').write(struct.pack('<%dI' % len(o), *o))
