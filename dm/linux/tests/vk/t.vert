#version 450
layout(set = 0, binding = 0) uniform support_buffer { vec4 s[4]; } sb;
layout(set = 0, binding = 1) uniform vp_c3 { vec4 c[4096]; } u;
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 0) out vec4 o0;
void main() { gl_Position = a0 + u.c[0] * 0.0; o0 = a1; }
