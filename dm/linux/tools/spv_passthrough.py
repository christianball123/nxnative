# NVIDIA passthrough geometry shaders (SPV_NV_geometry_shader_passthrough + SPV_NV_viewport_array2) → ordinary geometry shaders, for GPUs without them (Adreno).
# The translator gives a Maxwell passthrough geometry shader that form: it computes per-primitive outputs (here the viewport mask, gl_ViewportMask[0]) and the
# hardware forwards the input vertices (members / variables decorated PassthroughNV) to every viewport in the mask. Rewritten: the mask goes to a Private variable,
# and at the shader's end the primitive is emitted once per set bit (16 viewports, unrolled), each vertex's passthrough values copied to outputs, gl_ViewportIndex = bit.
# Mario Tennis Fever's shadow cascades draw this way (one draw, up to 4 viewports).
import struct

CAP_PASSTHROUGH, CAP_VPMASK, CAP_MULTIVIEWPORT = 5251, 5255, 57
DEC_BUILTIN, DEC_LOCATION, DEC_COMPONENT, DEC_PASSTHROUGH = 11, 30, 31, 5250
BI_VPMASK, BI_VPINDEX, BI_LAYER = 5253, 10, 9
SC_INPUT, SC_OUTPUT, SC_PRIVATE = 1, 3, 6
IN_VERTS = {19: 1, 20: 2, 21: 4, 22: 3, 23: 6}            # input primitive execution mode → vertices
OUT_VERTS = {27: 1, 28: 2, 29: 3}                        # output primitive → vertices per emitted primitive
NVP = 16

def _ins(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0: raise ValueError('bad SPIR-V')
        yield i, words[i] & 0xffff, words[i:i + n]
        i += n

def _str(ops):
    b = struct.pack(f'<{len(ops)}I', *ops)
    return b.split(b'\0', 1)[0].decode()

def is_passthrough(words):
    return any(op == 17 and ins[1] == CAP_PASSTHROUGH for _, op, ins in _ins(words))

def rewrite(words):
    words = list(words)
    bound = words[3]
    def new_id():
        nonlocal bound
        bound += 1
        return bound - 1
    types, ptrs, consts, decos, mdecos, vars_, entry = {}, {}, {}, {}, {}, {}, None
    in_prim, out_prim = None, None
    for i, op, ins in _ins(words):
        if op == 15: entry = ins
        elif op == 16:
            if ins[2] in IN_VERTS: in_prim = ins[2]
            if ins[2] in OUT_VERTS: out_prim = ins[2]
        elif op in (19, 20, 21, 22, 23, 24, 28, 29, 30): types[ins[1]] = (op, list(ins[2:]))
        elif op == 32: ptrs[ins[1]] = (ins[2], ins[3])
        elif op == 43: consts[ins[2]] = (ins[1], ins[3])
        elif op == 71: decos.setdefault(ins[1], []).append(list(ins[2:]))
        elif op == 72: mdecos.setdefault(ins[1], []).append(list(ins[2:]))
        elif op == 59: vars_[ins[2]] = (ins[1], ins[3])
    if entry is None or in_prim is None or out_prim is None: raise ValueError('not a geometry shader')
    fn = entry[2]
    nin, nout = IN_VERTS[in_prim], OUT_VERTS[out_prim]
    if nin < nout: raise ValueError('fewer input vertices than an output primitive has')
    def builtin(v): return next((d[1] for d in decos.get(v, []) if d[0] == DEC_BUILTIN), None)
    def pointee(v): return ptrs[vars_[v][0]][1]
    mask_var = next((v for v in vars_ if builtin(v) == BI_VPMASK), None)
    if mask_var is None: raise ValueError('no viewport mask output')
    # the input and output gl_PerVertex blocks, and the passthrough user inputs (Location)
    in_block = out_block = None
    for v, (pt, sc) in vars_.items():
        t = ptrs[pt][1]
        if sc == SC_INPUT and types.get(t, (0,))[0] == 28:
            el = types[t][1][0]
            if types.get(el, (0,))[0] == 30 and any(m[1] == DEC_BUILTIN for m in mdecos.get(el, [])): in_block = (v, el)
        if sc == SC_OUTPUT and types.get(t, (0,))[0] == 30 and any(m[1] == DEC_BUILTIN for m in mdecos.get(t, [])): out_block = (v, t)
    user_in = [v for v, (pt, sc) in vars_.items() if sc == SC_INPUT and any(d[0] == DEC_PASSTHROUGH for d in decos.get(v, []))
               and any(d[0] == DEC_LOCATION for d in decos.get(v, []))]
    int_t = next((t for t, (op, a) in types.items() if op == 21 and a == [32, 1]), None)
    bool_t = next((t for t, (op, a) in types.items() if op == 20), None)
    new_types, new_globals, new_decos = [], [], []
    def ins_(op, *ops): return [(len(ops) + 1) << 16 | op, *ops]
    if int_t is None: int_t = new_id(); new_types.append(ins_(21, int_t, 32, 1))
    if bool_t is None: bool_t = new_id(); new_types.append(ins_(20, bool_t))
    cache = {}
    def const(v):
        if ('c', v) not in cache: c = new_id(); new_types.append(ins_(43, int_t, c, v & 0xffffffff)); cache[('c', v)] = c
        return cache[('c', v)]
    def ptr(sc, t):
        if ('p', sc, t) not in cache: p = new_id(); new_types.append(ins_(32, p, sc, t)); cache[('p', sc, t)] = p
        return cache[('p', sc, t)]
    # the mask: Private from now on (every pointer derived from it retyped)
    retype = {}
    mask_ptr_old = vars_[mask_var][0]
    retype[mask_var] = ptr(SC_PRIVATE, pointee(mask_var))
    # gl_ViewportIndex output
    vpi = new_id(); new_globals.append(ins_(59, ptr(SC_OUTPUT, int_t), vpi, SC_OUTPUT)); new_decos.append(ins_(71, vpi, DEC_BUILTIN, BI_VPINDEX))
    # outputs for the passthrough user inputs, at their locations
    user_out = []
    for v in user_in:
        arr = pointee(v); el = types[arr][1][0]
        o = new_id(); new_globals.append(ins_(59, ptr(SC_OUTPUT, el), o, SC_OUTPUT))
        for d in decos.get(v, []):
            if d[0] in (DEC_LOCATION, DEC_COMPONENT) or d[0] in (13, 14, 15, 16, 17): new_decos.append(ins_(71, o, *d))   # location, component, interpolation qualifiers
        user_out.append((v, o, el))
    # the code emitted before each OpReturn of the entry function
    def emit_code():
        code = []
        p = new_id(); code.append(ins_(65, ptr(SC_PRIVATE, int_t), p, mask_var, const(0)))
        mask = new_id(); code.append(ins_(61, int_t, mask, p))
        for b in range(NVP):
            bit = new_id(); code.append(ins_(199, int_t, bit, mask, const(1 << b)))
            cond = new_id(); code.append(ins_(171, bool_t, cond, bit, const(0)))
            then, merge = new_id(), new_id()
            code.append(ins_(247, merge, 0)); code.append(ins_(250, cond, then, merge)); code.append(ins_(248, then))
            for vtx in range(nout):
                if in_block and out_block:
                    iv, iel = in_block; ov, ot = out_block
                    for m, mt in enumerate(types[ot][1]):
                        ipt = types[iel][1][m] if m < len(types[iel][1]) else None
                        if ipt is None: continue
                        src = new_id(); code.append(ins_(65, ptr(SC_INPUT, ipt), src, iv, const(vtx), const(m)))
                        val = new_id(); code.append(ins_(61, ipt, val, src))
                        dst = new_id(); code.append(ins_(65, ptr(SC_OUTPUT, mt), dst, ov, const(m)))
                        code.append(ins_(62, dst, val))
                for v, o, el in user_out:
                    src = new_id(); code.append(ins_(65, ptr(SC_INPUT, el), src, v, const(vtx)))
                    val = new_id(); code.append(ins_(61, el, val, src)); code.append(ins_(62, o, val))
                code.append(ins_(62, vpi, const(b)))
                code.append(ins_(218))   # OpEmitVertex
            code.append(ins_(219))       # OpEndPrimitive
            code.append(ins_(249, merge)); code.append(ins_(248, merge))
        return code
    code = [x for c in emit_code() for x in c]
    def deps(t):
        o = t[0] & 0xffff
        return [t[3]] if o == 32 else [t[1]] if o == 43 else []
    def defined_id(o, ins):
        if o in (19, 20, 21, 22, 23, 24, 28, 29, 30, 32): return ins[1]
        if o in (43, 41, 42, 44, 59): return ins[2]
        return None
    pending = list(new_types); defined = set()
    def flush(dst):
        nonlocal pending
        progress = True
        while progress:
            progress = False
            for t in list(pending):
                if all(d in defined for d in deps(t)):
                    dst += t; pending.remove(t); defined.add(t[1] if (t[0] & 0xffff) != 43 else t[2]); progress = True
    # rebuild
    out = words[:5]; in_fn = False; types_done = False; deco_done = False; cap_done = False
    has_mv = any(w[1] == CAP_MULTIVIEWPORT for _, o2, w in _ins(words) if o2 == 17)
    derived = {mask_var}
    for i, op, ins in _ins(words):
        ins = list(ins)
        if op == 17 and ins[1] in (CAP_PASSTHROUGH, CAP_VPMASK): continue
        if op == 10 and _str(ins[1:]) in ('SPV_NV_geometry_shader_passthrough', 'SPV_NV_viewport_array2'): continue
        if op == 72 and len(ins) > 3 and ins[3] == DEC_PASSTHROUGH: continue
        if op == 71 and ins[2] == DEC_PASSTHROUGH: continue
        if op == 71 and ins[1] == mask_var and ins[2] == DEC_BUILTIN: continue
        if op != 17 and not cap_done:   # MultiViewport (gl_ViewportIndex in a geometry shader) after the other capabilities
            if not has_mv: out += ins_(17, CAP_MULTIVIEWPORT)
            cap_done = True
        if op == 15:   # the entry point's interface: the mask is no longer an output, the new outputs are
            nm_words = (len(_str(ins[3:]).encode()) // 4) + 1
            iface = [x for x in ins[3 + nm_words:] if x != mask_var] + [vpi] + [o for _, o, _ in user_out]
            ins = ins_(15, ins[1], ins[2], *ins[3:3 + nm_words], *iface)
        if op == 16 and ins[2] == 26: ins = ins_(16, ins[1], 26, nout * NVP)   # OutputVertices: one primitive per viewport
        if op == 54 and not types_done:   # the first function: the new variables go before it
            flush(out)
            if pending: raise ValueError('a new type refers to an undefined id')
            for g in new_globals: out += g
            types_done = True
        if op == 59 and ins[2] == mask_var:   # its Private pointer type must exist first
            ins[1] = retype[mask_var]; ins[3] = SC_PRIVATE
            flush(out)
        if op in (65, 66, 83) and len(ins) > 3 and ins[3] in derived:   # pointers derived from the mask: Private
            derived.add(ins[2]); pt = ins[1]
            if pt in ptrs and ptrs[pt][0] == SC_OUTPUT: ins[1] = ptr(SC_PRIVATE, ptrs[pt][1])
        if op == 54: in_fn = ins[2] == fn
        if op == 253 and in_fn: out += code
        out += ins
        did = defined_id(op, ins)
        if did is not None and not in_fn: defined.add(did); flush(out)
        if op == 71 and not deco_done and new_decos:   # the new decorations right after the first decoration
            for d in new_decos: out += d
            deco_done = True
    if not types_done: raise ValueError('no function')
    out[3] = bound
    return out

def _fix_late_types(out, new_types):
    have = set()
    for _, op, ins in _ins(out):
        if op in (32, 43, 20, 21): have.add(ins[1] if op != 43 else ins[2])
    late = [t for t in new_types if (t[1] if (t[0] & 0xffff) != 43 else t[2]) not in have]
    if not late: return out
    res = out[:5]; placed = False
    for _, op, ins in _ins(out):
        if op == 54 and not placed:
            for t in late: res += t
            placed = True
        res += list(ins)
    return res
