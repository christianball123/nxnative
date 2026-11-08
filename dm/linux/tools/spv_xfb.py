# Transform feedback on Vulkan (VK_EXT_transform_feedback): a vertex shader whose program captures outputs (dmsh_build.py tf_layout, the job's 'tf': per feedback buffer its stride and
# the outputs it takes, [location, components], interleaved in that order) gets the SPIR-V decorations that capture them: capability TransformFeedback, execution mode Xfb, and on each
# captured output XfbBuffer, XfbStride and Offset. An output captured whole (4 components) is decorated as it is; one captured in part, captured twice, or not written by the
# shader goes through an output variable of its own at a free location, given the output's first components (or Maxwell's (0, 0, 0, 1)) before every return of the entry function.
# Pokémon Legends: Z-A skins its characters this way: a vertex-only program under rasterizer discard writes the skinned vertices, which the next draws read as a vertex buffer.
# The Mac's way (dmsh_build.py vs_tf: the vertex function stores into buffers) needs no extension; on Linux the replay binds the buffers (gfx_vk_draw.inc tf_*).
import struct

def _ins(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0: raise ValueError('bad SPIR-V')
        yield i, words[i] & 0xffff, list(words[i:i + n])
        i += n

def _op(op, *a): return [(len(a) + 1) << 16 | op, *a]

def add_xfb(words, tf):
    """words: a vertex shader's SPIR-V; tf: the job's 'tf' (4 entries, empty or {'stride', 'outputs'}). → (words, strides) with strides the 4 buffers' strides (0: none)"""
    if not tf or not any(tf): return list(words), [0, 0, 0, 0]
    words = list(words)
    caps, entry, ep_at, types, ptrs, consts, decos, outs = set(), None, None, {}, {}, {}, {}, {}
    first_fn = None; last_cap = last_ep = last_mode = last_deco = None; ret_at = []; in_entry = False; bound = words[3]
    for i, op, ins in _ins(words):
        if op == 17: caps.add(ins[1]); last_cap = i
        elif op == 15:
            last_ep = i
            if ins[1] == 0 and entry is None: entry, ep_at = ins[2], i
        elif op in (16, 331): last_mode = i
        elif op in (71, 72, 73, 74, 75, 332): last_deco = i; decos.setdefault(ins[1], []).append(ins[2:])
        elif op in (22, 23): types[ins[1]] = (op, ins[2:])
        elif op == 32: ptrs[(ins[2], ins[3])] = ins[1]
        elif op == 43 and len(ins) == 4: consts[(ins[1], ins[3])] = ins[2]
        elif op == 59 and ins[3] == 3: outs[ins[2]] = ins[1]
        elif op == 54:
            if first_fn is None: first_fn = i
            in_entry = ins[2] == entry
        elif op == 56: in_entry = False
        elif op == 253 and in_entry: ret_at.append(i)
    if entry is None: raise ValueError('no vertex entry point')
    if not ret_at: raise ValueError('the entry function has no OpReturn')
    ptr_type = {pid: pt for (sc, pt), pid in ptrs.items() if sc == 3}
    by_loc = {}
    for v, pt in outs.items():
        d = decos.get(v, [])
        if any(x[0] == 11 for x in d): continue   # built-ins (position, point size, clip distances) have no location
        loc = next((x[1] for x in d if x[0] == 30), None)
        if loc is not None: by_loc[loc] = (v, ptr_type.get(pt))
    f32 = next((t for t, (o, a) in types.items() if o == 22 and a == [32]), None)
    v4 = next((t for t, (o, a) in types.items() if o == 23 and a == [f32, 4]), None) if f32 is not None else None

    new_types, new_decos, new_vars, body = [], [], [], []   # body: what goes before each OpReturn
    def nid():
        nonlocal bound
        bound += 1; return bound - 1
    def float_type():
        nonlocal f32
        if f32 is None: f32 = nid(); new_types.extend(_op(22, f32, 32))
        return f32
    def vec_type(n):
        if n == 1: return float_type()
        f = float_type()
        t = next((t for t, (o, a) in types.items() if o == 23 and a == [f, n]), None)
        if t is None: t = nid(); types[t] = (23, [f, n]); new_types.extend(_op(23, t, f, n))
        return t
    def out_ptr(t):
        p = ptrs.get((3, t))
        if p is None: p = nid(); ptrs[(3, t)] = p; new_types.extend(_op(32, p, 3, t))
        return p
    def fconst(x):
        f = float_type(); bits = struct.unpack('<I', struct.pack('<f', x))[0]
        c = consts.get((f, bits))
        if c is None: c = nid(); consts[(f, bits)] = c; new_types.extend(_op(43, f, c, bits))
        return c
    used = set(by_loc); free = [l for l in range(31, -1, -1) if l not in used]
    taken, strides, captured = set(), [0, 0, 0, 0], 0
    for b, e in enumerate(tf[:4]):
        if not e: continue
        strides[b] = e['stride']; off = 0
        for loc, n in e['outputs']:
            src = by_loc.get(loc)
            if src and src[1] == v4 and n == 4 and src[0] not in taken:   # the output itself
                taken.add(src[0])
                new_decos += _op(71, src[0], 36, b) + _op(71, src[0], 37, e['stride']) + _op(71, src[0], 35, 4 * off)
            else:   # an output variable of its own, filled before every return
                if not free: raise ValueError('no free output location for a transform feedback copy')
                t = vec_type(n); p = out_ptr(t); var = nid(); l = free.pop(0)
                new_vars += _op(59, p, var, 3)
                new_decos += _op(71, var, 30, l) + _op(71, var, 36, b) + _op(71, var, 37, e['stride']) + _op(71, var, 35, 4 * off)
                body.append((src[0] if src and src[1] == v4 else None, t, n, var))
            off += n; captured += 1
    if not captured: return words, [0, 0, 0, 0]
    # constants for outputs the shader does not write, made once
    defaults = {}
    for s, t, n, var in body:
        if s is None and t not in defaults:
            comps = [fconst(0.0)] * min(n, 3) + ([fconst(1.0)] if n == 4 else [])
            if n == 1: defaults[t] = comps[0]
            else: c = nid(); new_types.extend(_op(44, t, c, *comps)); defaults[t] = c
    v4_needed = any(s is not None for s, _, _, _ in body)
    out = list(words[:5]); out[3] = 0   # the bound is set at the end
    for i, op, ins in _ins(words):
        if i == first_fn: out += new_types + new_vars
        if op == 15 and i == ep_at: ins = ins + [v for _, _, _, v in body]; ins[0] = len(ins) << 16 | 15
        if i in ret_at:
            for s, t, n, var in body:
                if s is None: out += _op(62, var, defaults[t]); continue   # OpStore
                l = nid(); out += _op(61, v4, l, s)                         # OpLoad
                if n == 4: out += _op(62, var, l)
                elif n == 1: x = nid(); out += _op(81, t, x, l, 0) + _op(62, var, x)   # OpCompositeExtract
                else: x = nid(); out += _op(79, t, x, l, l, *range(n)) + _op(62, var, x)   # OpVectorShuffle
        out += ins
        if i == last_cap and 53 not in caps: out += _op(17, 53)            # OpCapability TransformFeedback
        if i == (last_mode if last_mode is not None else last_ep): out += _op(16, entry, 11)   # OpExecutionMode Xfb
        if i == last_deco: out += new_decos
    if last_deco is None: raise ValueError('no decorations')
    if v4_needed and v4 is None: raise ValueError('outputs are not vec4')
    out[3] = bound
    return out, strides
