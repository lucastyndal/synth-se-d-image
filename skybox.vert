#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection; 

mat4 view_matrix_no_translation;

in vec3 position;
in vec2 tex_coord;

out vec2 frag_tex_coords;

void main() {
    // Suppression du composant de translation
    view_matrix_no_translation = view;
    view_matrix_no_translation[3][0] = 0;
    view_matrix_no_translation[3][1] = 0;
    view_matrix_no_translation[3][2] = 0;


    gl_Position = projection * view_matrix_no_translation * model * vec4(position, 1);
    frag_tex_coords = tex_coord;
}
