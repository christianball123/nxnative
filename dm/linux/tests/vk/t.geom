#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(set = 0, binding = 0) uniform support_buffer { vec4 s[4]; } sb;
layout(location = 0) in vec4 i0[];
layout(location = 0) out vec4 o0;
void main() { for (int k = 0; k < 3; k++) { gl_Position = gl_in[k].gl_Position; o0 = i0[k]; EmitVertex(); } EndPrimitive(); }
