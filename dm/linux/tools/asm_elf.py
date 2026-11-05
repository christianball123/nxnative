#!/usr/bin/env python3
# The host's assembly (dm/host/entry.S, dm/gfx/nvn/*.S) is written for Mach-O: C symbols carry a leading underscore, page addressing is sym@PAGE / sym@PAGEOFF,
# and .zerofill / .cstring name Mach-O sections. This writes the ELF form of a file, for the Linux build: the same instructions, ELF spelling.
# Usage: asm_elf.py <in.S> <out.S>
import re, sys
src, dst = sys.argv[1], sys.argv[2]
out = []
sym = re.compile(r'(?<![\w.$])_([A-Za-z][\w]*)')   # _name -> name (not __name, not a local .L label)
for line in open(src):
    code, sep, comment = line.partition('//')
    m = re.match(r'(\s*)\.zerofill\s+[^,]+,[^,]+,\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*(\w+))?', code)
    if m:
        ind, name, size, align = m.group(1), sym.sub(r'\1', m.group(2)), m.group(3), m.group(4) or '0'
        out.append(f'{ind}.pushsection .bss\n{ind}.balign {1 << int(align, 0)}\n{name}:\n{ind}.zero {size}\n{ind}.popsection{"  //" + comment.rstrip() if sep else ""}\n')
        continue
    if re.match(r'\s*\.cstring\b', code):
        out.append(code.replace('.cstring', '.section .rodata.str1.1, "aMS", @progbits, 1') + (sep + comment if sep else '\n'))
        continue
    code = re.sub(r'(\w+)@PAGEOFF', r':lo12:\1', code)
    code = re.sub(r'(\w+)@PAGE\b', r'\1', code)
    code = sym.sub(r'\1', code)
    out.append(code + (sep + comment if sep else ''))
open(dst, 'w').write(''.join(out) + '\n    .section .note.GNU-stack, "", @progbits\n')
