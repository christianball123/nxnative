#version 450
layout(set = 0, binding = 0, std140) uniform support_buffer { uint a; uint b[8]; vec4 c; vec4 d; int e; float rs[73]; ivec4 f; int g; } sb;   // the translator's support buffer layout
layout(set = 0, binding = 1) uniform sampler2D tex;
layout(set = 0, binding = 2) uniform fp_c3 { vec4 c[4096]; } u;
layout(location = 0) in vec4 i0;
layout(location = 0) out vec4 col;
void main() { col = texture(tex, i0.xy) * u.c[0] + vec4(0, 0, 0, 0) * sb.rs[0]; }
