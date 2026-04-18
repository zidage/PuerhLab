//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>
#include <libraw/libraw.h>

namespace alcedo {
namespace raw_norm {

#ifdef __CUDACC__
#define ALCEDO_RAW_NORM_HD __host__ __device__
#else
#define ALCEDO_RAW_NORM_HD
#endif

struct RawLinearizationCurve {
  std::array<float, 4> black_level = {};
  std::array<float, 4> white_level = {};
};

ALCEDO_RAW_NORM_HD inline auto Clamp01(const float value) -> float {
  return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

ALCEDO_RAW_NORM_HD inline auto NormalizeSample(const float sample, const float black,
                                                 const float white) -> float {
  const float denom = white - black;
  if (!(denom > 0.0f)) {
    return 0.0f;
  }
  return Clamp01((sample - black) / denom);
}

ALCEDO_RAW_NORM_HD inline auto RelativeWhiteBalanceMultiplier(const float* wb_multipliers,
                                                                const int color_idx,
                                                                const bool apply_white_balance)
    -> float {
  if (!apply_white_balance) {
    return 1.0f;
  }

  const float green = wb_multipliers[1];
  if (!(green > 0.0f)) {
    return 1.0f;
  }

  return (color_idx == 0 || color_idx == 2) ? (wb_multipliers[color_idx] / green) : 1.0f;
}

inline auto BuildLinearizationCurve(const libraw_rawdata_t& raw_data) -> RawLinearizationCurve {
  RawLinearizationCurve curve = {};
  const float           base_black = static_cast<float>(raw_data.color.black);
  const float           fallback_white = static_cast<float>(raw_data.color.maximum);

  for (int c = 0; c < 4; ++c) {
    curve.black_level[c] = base_black + static_cast<float>(raw_data.color.cblack[c]);

    const float channel_white = static_cast<float>(raw_data.color.linear_max[c]);
    curve.white_level[c]      = channel_white > 0.0f ? channel_white : fallback_white;
  }

  return curve;
}

inline auto PatternBlackAt(const libraw_rawdata_t& raw_data, const int y, const int x) -> float {
  const int tile_width  = raw_data.color.cblack[4];
  const int tile_height = raw_data.color.cblack[5];
  if (tile_width <= 0 || tile_height <= 0) {
    return 0.0f;
  }

  const int tile_y = ((y % tile_height) + tile_height) % tile_height;
  const int tile_x = ((x % tile_width) + tile_width) % tile_width;
  return static_cast<float>(raw_data.color.cblack[6 + tile_y * tile_width + tile_x]);
}

#undef ALCEDO_RAW_NORM_HD

}  // namespace raw_norm
}  // namespace alcedo
