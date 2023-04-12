#version 330 core

// input from vertex shader
in vec3 w_position, w_normal;
in vec2 frag_tex_coords;
in float fogIntensity;
in float fogDensity;

// uniform variables
uniform vec3 camera_position;
uniform sampler2D diffuse_map;

void main() {
    const vec3 fogColor = vec3(0.5,0.5,0.5);

    // calculate distance between fragment and camera
    float d = length(camera_position - w_position);
    
    // calculate fog density using distance and density factor
    float fogDensity = exp(-fogIntensity * fogDensity * d * d);
    
    // sample texture and apply fog density
    vec4 color = texture(diffuse_map, frag_tex_coords);
    color.rgb = mix(color.rgb, fogColor.rgb, 1-fogDensity);
    
    gl_FragColor = color;
}