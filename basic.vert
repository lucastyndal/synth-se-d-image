#version 330 core

// input attribute variable, given per vertex
in vec3 position;
in vec3 normal;
in vec2 tex_coord;

uniform mat4 model, view, projection, rotations;

// position and normal for the fragment shader, in WORLD coordinates
out vec3 w_position, w_normal;   // in world coordinates

// Tex coords
out vec2 frag_tex_coords;

// Fog
out float fogIntensity;
out float fogDensity;

void main() {
    fogDensity = 0.0004;

    vec4 w_position4 = model * vec4(position, 1.0);
    gl_Position = projection * view * w_position4;

    // fragment position in world coordinates
    w_position = w_position4.xyz / w_position4.w;  // dehomogenize

    // fragment normal in world coordinates
    mat3 nit_matrix = transpose(inverse(mat3(model)));
    w_normal = normalize(nit_matrix * normal);

    // textures
    frag_tex_coords = tex_coord;

    // Fog
    fogIntensity = exp(-gl_Position.z * fogDensity * fogDensity);
}