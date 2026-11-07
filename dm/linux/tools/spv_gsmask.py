# A passthrough geometry shader's viewport mask, when it comes from uniforms alone (Mario Tennis Fever's shadow cascades: a per-draw mask read from c5 and c10),
# as a little postfix program the host runs for each draw with the bound uniform buffers (gfx_vk_draw.inc gs_mask): the draw then goes without the geometry
# shader, once to each viewport in the mask. Geometry shaders are slow on Adreno: these draws took half of a match frame's GPU time.
# Program words: 1 v: push v | 2 slot: pop a dword index, push that dword of the shader's c[slot] | 3 add 4 sub 5 mul 6 and 7 or 8 xor 9 shl 10 shr (logical)
# 11 sar: pop b, pop a, push a op b. None when the mask depends on anything else (inputs, branches, other memory).
OPS = {128: 3, 130: 4, 132: 5, 199: 6, 197: 7, 198: 8, 196: 9, 194: 10, 195: 11}   # OpIAdd OpISub OpIMul OpBitwiseAnd/Or/Xor OpShiftLeftLogical/RightLogical/RightArithmetic

def _ins(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0: raise ValueError('bad SPIR-V')
        yield words[i] & 0xffff, words[i:i + n]
        i += n

def mask_program(words, cbufs):
    """cbufs: binding -> cbuf slot (the shader's res.json). Returns a list of u32 or None."""
    consts, vars_, deco, ptr_sc = {}, {}, {}, {}
    entry = None
    for op, ins in _ins(words):
        if op == 15: entry = ins[2]
        elif op == 43: consts[ins[2]] = ins[3]
        elif op == 59: vars_[ins[2]] = ins[3]
        elif op == 71: deco.setdefault(ins[1], []).append(list(ins[2:]))
    def binding(v): return next((d[1] for d in deco.get(v, []) if d[0] == 33), None)
    mask_vars = {v for v, ds in deco.items() if any(d[0] == 11 and d[1] == 5253 for d in ds)}
    if not mask_vars: return None
    env = {}; mem = {}; chains = {}; in_main = False; labels = 0
    def leaf_const(v): return ['c', v]
    for op, ins in _ins(words):
        if op == 54: in_main = ins[2] == entry; continue
        if not in_main: continue
        if op == 248:
            labels += 1
            if labels > 1: return None   # more than one block: not straight-line
            continue
        if op in (59, 8): continue      # OpVariable, OpLine
        if op == 65:                    # OpAccessChain
            chains[ins[2]] = (ins[3], list(ins[4:])); continue
        if op == 61:                    # OpLoad
            p = ins[3]
            if p in mem: env[ins[2]] = mem[p]; continue
            if p in chains:
                base, idx = chains[p]
                if vars_.get(base) != 2 or binding(base) is None or binding(base) not in cbufs or len(idx) != 3: return None
                if idx[0] not in consts or consts[idx[0]] != 0: return None
                def val(i): return ['c', consts[i]] if i in consts else env.get(i)
                a, b = val(idx[1]), val(idx[2])
                if a is None or b is None: return None
                env[ins[2]] = ['load', cbufs[binding(base)], ['op', 3, ['op', 5, a, ['c', 4]], b]]; continue
            return None                 # anything else (inputs, other memory)
        if op == 62:                    # OpStore
            p, v = ins[1], ins[2]
            val = env.get(v) if v not in consts else ['c', consts[v]]
            tgt = chains[p][0] if p in chains else p
            if tgt in mask_vars:
                if val is None: return None
                out = []
                def emit(e):
                    if e[0] == 'c': out.extend([1, e[1] & 0xffffffff])
                    elif e[0] == 'load': emit(e[2]); out.extend([2, e[1]])
                    else: emit(e[2]); emit(e[3]); out.append(e[1])
                emit(val); return out
            if val is None: return None
            mem[p] = val; continue
        if op in (124, 113, 114, 83):   # OpBitcast, OpUConvert, OpSConvert, OpCopyObject: the same bits
            src = ins[3]; env[ins[2]] = ['c', consts[src]] if src in consts else env.get(src)
            if env[ins[2]] is None: return None
            continue
        if op in OPS:
            a = ['c', consts[ins[3]]] if ins[3] in consts else env.get(ins[3])
            b = ['c', consts[ins[4]]] if ins[4] in consts else env.get(ins[4])
            if a is None or b is None: return None
            env[ins[2]] = ['op', OPS[op], a, b]; continue
        if op in (253, 56): return None   # returned without storing the mask
        return None                       # an operation this does not follow
    return None

if __name__ == '__main__':
    import sys, struct, json
    b = open(sys.argv[1], 'rb').read(); w = list(struct.unpack('<%dI' % (len(b) // 4), b))
    res = json.load(open(sys.argv[2]))
    print(mask_program(w, {c['binding']: c['slot'] for c in res.get('cbufs', [])}))
