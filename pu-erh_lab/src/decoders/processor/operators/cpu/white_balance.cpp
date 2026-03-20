//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// FIXME: What this file does is not just white balance, but also black level correction and
// scaling. Should rename it to something more appropriate.

#include "decoders/processor/operators/cpu/white_balance.hpp"

#include <cfloat>

namespace puerhlab {
namespace CPU {
static auto CalculateBlackLevel(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 4> black_level      = {
      base_black_level + static_cast<float>(raw_data.color.cblack[0]),
      base_black_level + static_cast<float>(raw_data.color.cblack[1]),
      base_black_level + static_cast<float>(raw_data.color.cblack[2]),
      base_black_level + static_cast<float>(raw_data.color.cblack[3])};

  return black_level;
}

static inline auto PatternBlackAt(const libraw_rawdata_t& raw_data, int y, int x) -> float {
  const int tile_width  = raw_data.color.cblack[4];
  const int tile_height = raw_data.color.cblack[5];
  if (tile_width <= 0 || tile_height <= 0) {
    return 0.0f;
  }
  const int tile_y = ((y % tile_height) + tile_height) % tile_height;
  const int tile_x = ((x % tile_width) + tile_width) % tile_width;
  return static_cast<float>(raw_data.color.cblack[6 + tile_y * tile_width + tile_x]) / 65535.0f;
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

inline static auto GetScaleMul(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  std::array<float, 4> scale_mul = {};
  for (int c = 0; c < 4; ++c) {
    scale_mul[c] = raw_data.color.cam_mul[c];
  }

  return scale_mul;
}

void ToLinearRef(cv::Mat& img, LibRaw& raw_processor) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);
  int  w           = img.cols;
  int  h           = img.rows;

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    for (float& level : black_level) {
      level /= 65535.0f;
    }
    const float maximum = raw_processor.imgdata.rawdata.color.maximum / 65535.0f - black_level[0];

    auto scale_mul = GetScaleMul(raw_processor.imgdata.rawdata);
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int   color_idx     = raw_processor.COLOR(y, x);

        float pixel         = img.at<float>(y, x);

        pixel -= black_level[color_idx] + PatternBlackAt(raw_processor.imgdata.rawdata, y, x);

        const float mask   = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
        const float wb_mul = (scale_mul[color_idx] / scale_mul[1]) * mask + (1.0f - mask);
        pixel *= wb_mul;
        pixel /= maximum;

        img.at<float>(y, x) = pixel;
      }
    }
  }
}

};  // namespace CPU
};  // namespace puerhlab
