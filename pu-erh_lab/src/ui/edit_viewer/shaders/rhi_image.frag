#version 440

layout(binding = 1) uniform sampler2D sourceTexture;

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 fragColor;

void main() {
  fragColor = texture(sourceTexture, vUv);
}
