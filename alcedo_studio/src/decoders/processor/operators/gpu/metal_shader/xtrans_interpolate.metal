//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct XTransParams {
  uint width;
  uint height;
  uint tile_width;
  uint tile_height;
  uint passes;
  uint green_radius;
  uint rb_radius;
  uint rgb_fc[36];
};

static inline int ClampCoord(int value, int limit) {
  return clamp(value, 0, limit - 1);
}

static inline uint PatternColorAt(constant XTransParams& params, int y, int x) {
  const int tile_h = max(1, static_cast<int>(params.tile_height));
  const int tile_w = max(1, static_cast<int>(params.tile_width));
  const int wrapped_y = (y % tile_h + tile_h) % tile_h;
  const int wrapped_x = (x % tile_w + tile_w) % tile_w;
  return params.rgb_fc[wrapped_y * tile_w + wrapped_x];
}

static inline float SafeRead(texture2d<float, access::read> tex, constant XTransParams& params,
                             int y, int x) {
  const uint2 coord(static_cast<uint>(ClampCoord(x, static_cast<int>(params.width))),
                    static_cast<uint>(ClampCoord(y, static_cast<int>(params.height))));
  return tex.read(coord).x;
}

static inline float FindDirectionalGreen(texture2d<float, access::read> raw,
                                         constant XTransParams& params, int y, int x) {
  float left  = SafeRead(raw, params, y, x);
  float right = left;
  float up    = left;
  float down  = left;

  bool has_left  = false;
  bool has_right = false;
  bool has_up    = false;
  bool has_down  = false;

  for (int radius = 1; radius <= static_cast<int>(params.green_radius) &&
                       (!has_left || !has_right);
       ++radius) {
    if (!has_left && PatternColorAt(params, y, x - radius) == 1u) {
      left     = SafeRead(raw, params, y, x - radius);
      has_left = true;
    }
    if (!has_right && PatternColorAt(params, y, x + radius) == 1u) {
      right     = SafeRead(raw, params, y, x + radius);
      has_right = true;
    }
  }

  for (int radius = 1; radius <= static_cast<int>(params.green_radius) &&
                       (!has_up || !has_down);
       ++radius) {
    if (!has_up && PatternColorAt(params, y - radius, x) == 1u) {
      up     = SafeRead(raw, params, y - radius, x);
      has_up = true;
    }
    if (!has_down && PatternColorAt(params, y + radius, x) == 1u) {
      down     = SafeRead(raw, params, y + radius, x);
      has_down = true;
    }
  }

  if (has_left && has_right && has_up && has_down) {
    const float horizontal_grad = fabs(left - right);
    const float vertical_grad   = fabs(up - down);
    return horizontal_grad <= vertical_grad ? 0.5f * (left + right) : 0.5f * (up + down);
  }
  if (has_left && has_right) {
    return 0.5f * (left + right);
  }
  if (has_up && has_down) {
    return 0.5f * (up + down);
  }

  float sum   = 0.0f;
  uint  count = 0u;
  for (int radius = 1; radius <= static_cast<int>(params.green_radius); ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (max(abs(dx), abs(dy)) != radius) {
          continue;
        }
        if (PatternColorAt(params, y + dy, x + dx) != 1u) {
          continue;
        }
        sum += SafeRead(raw, params, y + dy, x + dx);
        ++count;
      }
    }
    if (count > 0u) {
      break;
    }
  }

  return count > 0u ? sum / static_cast<float>(count) : SafeRead(raw, params, y, x);
}

static inline float EstimateMissingChannel(texture2d<float, access::read> raw,
                                           texture2d<float, access::read> green,
                                           constant XTransParams& params, int y, int x,
                                           uint target_color, float current_green) {
  float sum   = 0.0f;
  float wsum  = 0.0f;

  for (int radius = 1; radius <= static_cast<int>(params.rb_radius); ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (max(abs(dx), abs(dy)) != radius) {
          continue;
        }
        if (PatternColorAt(params, y + dy, x + dx) != target_color) {
          continue;
        }

        const float neigh_raw   = SafeRead(raw, params, y + dy, x + dx);
        const float neigh_green = SafeRead(green, params, y + dy, x + dx);
        const float weight      = 1.0f / static_cast<float>(abs(dx) + abs(dy));
        sum += (neigh_raw - neigh_green) * weight;
        wsum += weight;
      }
    }
    if (wsum > 0.0f) {
      break;
    }
  }

  if (wsum == 0.0f) {
    return current_green;
  }

  return max(0.0f, current_green + sum / wsum);
}

kernel void xtrans_green(texture2d<float, access::read>  raw [[texture(0)]],
                         texture2d<float, access::write> green [[texture(1)]],
                         constant XTransParams& params [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const int x = static_cast<int>(gid.x);
  const int y = static_cast<int>(gid.y);
  const uint color = PatternColorAt(params, y, x);
  const float value = color == 1u ? raw.read(gid).x : FindDirectionalGreen(raw, params, y, x);
  green.write(float4(value), gid);
}

kernel void xtrans_rgba(texture2d<float, access::read>   raw [[texture(0)]],
                        texture2d<float, access::read>   green [[texture(1)]],
                        texture2d<float, access::write>  rgba [[texture(2)]],
                        constant XTransParams& params [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  const int x = static_cast<int>(gid.x);
  const int y = static_cast<int>(gid.y);
  const uint color = PatternColorAt(params, y, x);
  const float raw_value = raw.read(gid).x;
  const float green_value = green.read(gid).x;

  float r = color == 0u ? raw_value : EstimateMissingChannel(raw, green, params, y, x, 0u, green_value);
  float g = green_value;
  float b = color == 2u ? raw_value : EstimateMissingChannel(raw, green, params, y, x, 2u, green_value);

  if (color == 1u) {
    g = raw_value;
  }

  rgba.write(float4(r, g, b, 1.0f), gid);
}
