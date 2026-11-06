# Maxwell gives a vertex shader's output attributes (0, 0, 0, 1) until the shader writes them: a component the shader leaves alone reaches the fragment shader as
# that default. In SPIR-V an unwritten output is undefined (Turnip passes 0), so every user output vec4 (a Location, float) of a vertex shader gets that value at
# the start of the entry function. Mario Tennis Fever's particle vertex shaders write only .x of one output whose .w the fragment shader uses as the texture's
# weight in its alpha: with 0 there instead of 1, the texture stopped shaping the effect and the quads drew solid.
import struct

def _ins(words):
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0: raise ValueError('bad SPIR-V')
        yield i, words[i] & 0xffff, words[i:i + n]
        i += n

def init_outputs(words):
    words = list(words)
    entry = None; types = {}; ptrs = {}; vars_ = {}; decos = {}
    for i, op, ins in _ins(words):
        if op == 15 and ins[1] == 0: entry = ins[2]              # OpEntryPoint Vertex
        elif op in (21, 22, 23): types[ins[1]] = (op, list(ins[2:]))
        elif op == 32: ptrs[ins[1]] = (ins[2], ins[3])
        elif op == 59: vars_[ins[2]] = (ins[1], ins[3])
        elif op == 71: decos.setdefault(ins[1], []).append(list(ins[2:]))
    if entry is None: return words, False
    f32 = next((t for t, (o, a) in types.items() if o == 22 and a == [32]), None)
    v4 = next((t for t, (o, a) in types.items() if o == 23 and f32 is not None and a == [f32, 4]), None)
    if v4 is None: return words, False
    outs = [v for v, (pt, sc) in vars_.items() if sc == 3 and pt in ptrs and ptrs[pt][1] == v4
            and any(d[0] == 30 for d in decos.get(v, [])) and not any(d[0] == 11 for d in decos.get(v, []))]
    if not outs: return words, False
    bound = words[3]
    c0, c1, cv = bound, bound + 1, bound + 2; bound += 3
    consts = [4 << 16 | 43, f32, c0, 0, 4 << 16 | 43, f32, c1, 0x3f800000, 7 << 16 | 44, v4, cv, c0, c0, c0, c1]
    out = list(words[:5]); state = 0; placed = False   # state: 1 in the entry function, 2 its first block's variables
    for i, op, ins in _ins(words):
        if op == 54 and not placed: out += consts; placed = True   # before the first function
        if op == 54: state = 1 if ins[2] == entry else 0
        elif state == 1 and op == 248: state = 2; out += list(ins); continue
        elif state == 2 and op != 59 and op != 8:   # past the OpVariables (and OpLines) of the first block
            for v in outs: out += [3 << 16 | 62, v, cv]
            state = 0
        out += list(ins)
    out[3] = bound
    return out, True

if __name__ == '__main__':
    import sys
    b = open(sys.argv[1], 'rb').read(); w = list(struct.unpack('<%dI' % (len(b) // 4), b))
    o, ok = init_outputs(w); print('done' if ok else 'not applied')
    if ok and len(sys.argv) > 2: open(sys.argv[2], 'wb').write(struct.pack('<%dI' % len(o), *o))
