#version 450
layout(set = 0, binding = 0) uniform support_buffer { vec4 s[4]; } sb;
layout(set = 0, binding = 1) uniform sampler2D tex;
layout(set = 0, binding = 2) uniform fp_c3 { vec4 c[4096]; } u;
layout(location = 0) in vec4 i0;
layout(location = 0) out vec4 col;
void main() { col = texture(tex, i0.xy) * u.c[0] + vec4(0, 0, 0, 0); }
