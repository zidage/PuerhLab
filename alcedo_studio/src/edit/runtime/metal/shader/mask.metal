//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct MaskAnalyticParams {
  float render_to_reference[9];
  uint  width;
  uint  height;
  uint  reference_width;
  uint  reference_height;
  uint  kind;
  float center_x;
  float center_y;
  float major_radius;
  float minor_radius;
  float rotation;
  float inner_feather;
  float outer_feather;
  uint  radial_invert;
  float origin_x;
  float origin_y;
  float normal_x;
  float normal_y;
  float transition_distance;
  float start_value;
  float end_value;
  float opacity;
};

static inline float2 Transform(constant float m[9], float x, float y) {
  return float2(m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
}

static inline float FinishEffectiveCoverage(float value, uint invert, float opacity) {
  if (invert != 0u) {
    value = 1.0f - value;
  }
  return min(max(value * opacity, 0.0f), 1.0f);
}

static inline float QuantizeR8(float value) {
  return float(uint(min(max(value * 255.0f + 0.5f, 0.0f), 255.0f))) / 255.0f;
}

kernel void mask_analytic(texture2d<float, access::write> output [[texture(0)]],
                          constant MaskAnalyticParams& params [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const float2 reference =
      Transform(params.render_to_reference, float(gid.x) + 0.5f, float(gid.y) + 0.5f);
  const float nx    = reference.x / static_cast<float>(params.reference_width);
  const float ny    = reference.y / static_cast<float>(params.reference_height);
  float       value = 0.0f;
  uint        invert = 0;
  if (params.kind == 0u) {
    const float c      = cos(params.rotation);
    const float s      = sin(params.rotation);
    const float dx     = nx - params.center_x;
    const float dy     = ny - params.center_y;
    const float rx     = (c * dx + s * dy) / max(params.major_radius, 1.0e-6f);
    const float ry     = (-s * dx + c * dy) / max(params.minor_radius, 1.0e-6f);
    const float radius = sqrt(rx * rx + ry * ry);
    const float inner  = max(0.0f, 1.0f - params.inner_feather);
    const float outer  = 1.0f + params.outer_feather;
    value  = 1.0f - min(max((radius - inner) / max(outer - inner, 1.0e-6f), 0.0f), 1.0f);
    invert = params.radial_invert;
  } else {
    const float normal_length = sqrt(params.normal_x * params.normal_x + params.normal_y * params.normal_y);
    const float normal_x      = params.normal_x / max(normal_length, 1.0e-6f);
    const float normal_y      = params.normal_y / max(normal_length, 1.0e-6f);
    const float distance =
        (nx - params.origin_x) * normal_x + (ny - params.origin_y) * normal_y;
    const float t =
        min(max(distance / max(params.transition_distance, 1.0e-6f) + 0.5f, 0.0f), 1.0f);
    value  = params.start_value + (params.end_value - params.start_value) * t;
    invert = params.radial_invert;
  }
  output.write(float4(QuantizeR8(FinishEffectiveCoverage(value, invert, params.opacity)), 0.0f,
                      0.0f, 1.0f),
               gid);
}

// Raster MaskEvaluate (SM6 R7). The matrix maps render pixels to texture UV.
struct MaskRasterParams {
  float render_to_texture_uv[9];
  uint  width;
  uint  height;
  uint  texture_width;
  uint  texture_height;
  uint  samples;
  uint  invert;
  float opacity;
};

// Bilinear read at texel centers with clamp-to-edge indices; 0 outside UV [0, 1].
static inline float SampleRasterMask(texture2d<float, access::read> texture,
                                     constant MaskRasterParams& params, float x, float y) {
  const float u = params.render_to_texture_uv[0] * x + params.render_to_texture_uv[1] * y +
                  params.render_to_texture_uv[2];
  const float v = params.render_to_texture_uv[3] * x + params.render_to_texture_uv[4] * y +
                  params.render_to_texture_uv[5];
  if (!(u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f)) {
    return 0.0f;
  }
  const float tx    = u * float(params.texture_width) - 0.5f;
  const float ty    = v * float(params.texture_height) - 0.5f;
  const float x0f   = floor(tx);
  const float y0f   = floor(ty);
  const float fx    = tx - x0f;
  const float fy    = ty - y0f;
  const int   max_x = int(params.texture_width) - 1;
  const int   max_y = int(params.texture_height) - 1;
  const uint  x0    = uint(clamp(int(x0f), 0, max_x));
  const uint  x1    = uint(clamp(int(x0f) + 1, 0, max_x));
  const uint  y0    = uint(clamp(int(y0f), 0, max_y));
  const uint  y1    = uint(clamp(int(y0f) + 1, 0, max_y));
  const float t00   = texture.read(uint2(x0, y0)).x;
  const float t10   = texture.read(uint2(x1, y0)).x;
  const float t01   = texture.read(uint2(x0, y1)).x;
  const float t11   = texture.read(uint2(x1, y1)).x;
  const float top    = t00 + (t10 - t00) * fx;
  const float bottom = t01 + (t11 - t01) * fx;
  return top + (bottom - top) * fy;
}

kernel void mask_raster(texture2d<float, access::read> texture [[texture(0)]],
                        texture2d<float, access::write> output [[texture(1)]],
                        constant MaskRasterParams& params [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  const float samples = float(params.samples);
  float       sum     = 0.0f;
  for (uint j = 0; j < params.samples; ++j) {
    for (uint i = 0; i < params.samples; ++i) {
      sum += SampleRasterMask(texture, params, float(gid.x) + (float(i) + 0.5f) / samples,
                              float(gid.y) + (float(j) + 0.5f) / samples);
    }
  }
  output.write(float4(QuantizeR8(FinishEffectiveCoverage(sum / (samples * samples), params.invert,
                                                          params.opacity)),
                      0.0f, 0.0f, 1.0f),
               gid);
}

kernel void mask_fill_zero(texture2d<float, access::write> output [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
  output.write(float4(0.0f, 0.0f, 0.0f, 1.0f), gid);
}

kernel void mask_union_max(texture2d<float, access::read> lhs [[texture(0)]],
                           texture2d<float, access::read> rhs [[texture(1)]],
                           texture2d<float, access::write> output [[texture(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const float a = lhs.read(gid).x;
  const float b = rhs.read(gid).x;
  output.write(float4(QuantizeR8(max(a, b)), 0.0f, 0.0f, 1.0f), gid);
}
