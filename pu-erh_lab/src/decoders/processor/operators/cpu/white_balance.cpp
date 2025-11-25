// FIXME: What this file does is not just white balance, but also black level correction and scaling.
// Should rename it to something more appropriate.

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

  if (raw_data.color.cblack[4] == 2 && raw_data.color.cblack[5] == 2) {
    for (unsigned int x = 0; x < raw_data.color.cblack[4]; ++x) {
      for (unsigned int y = 0; y < raw_data.color.cblack[5]; ++y) {
        const auto index   = y * 2 + x;
        black_level[index] = raw_data.color.cblack[6 + index];
      }
    }
  }

  return black_level;
}

static auto GetWBCoeff(const libraw_rawdata_t& raw_data) -> const float* {
  return raw_data.color.cam_mul;
}

inline static auto GetScaleMul(const libraw_rawdata_t& raw_data) -> std::array<float, 4> {
  // cam_mul for as-shot white balance, pre_mul for D65
  auto                 cam_mul = raw_data.color.cam_mul;
  auto                 pre_mul = raw_data.color.pre_mul;

  auto                 c_white = (int)raw_data.color.maximum;
  auto                 c_black = (int)raw_data.color.black;

  // From dcraw.c

  std::array<float, 4> scale_mul;
  for (int c = 0; c < 4; ++c) {
    float mul_c = cam_mul[c];  
    if (mul_c == 0.f) {
      mul_c = cam_mul[1];
    }

    scale_mul[c] = (mul_c / cam_mul[1]) / ((c_white - c_black) / 65535.0f);
  }

  return scale_mul;
}

void WhiteBalanceCorrection(cv::Mat& img, LibRaw& raw_processor) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);
  int  w           = img.cols;
  int  h           = img.rows;

  if (raw_processor.imgdata.color.as_shot_wb_applied != 1) {
    for (float& level : black_level) {
      level /= 65535.0f;
    }
    float min            = black_level[0];
    auto  linear_maximum = raw_processor.imgdata.rawdata.color.linear_max;
    float maximum[4];
    for (int i = 0; i < 4; ++i) {
      if (linear_maximum[i] == 0) {
        maximum[i] = raw_processor.imgdata.rawdata.color.maximum / 65535.0f - min;
      } else {
        maximum[i] = linear_maximum[i] / 65535.0f - min;
      }
    }

    auto scale_mul = GetScaleMul(raw_processor.imgdata.rawdata);
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int   color_idx     = raw_processor.COLOR(y, x);

        float pixel         = img.at<float>(y, x);

        pixel               = std::max(0.0f, pixel - black_level[color_idx]);
        pixel               = pixel * scale_mul[color_idx];

        // //
        // pixel               = std::min(1.2f, pixel);
        pixel               = std::max(0.0f, pixel);
        // float muled_pixel = pixel;
        // float mask        = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
        // //
        // float wb_mul      = (wb[color_idx] / wb[1]) * mask + (1.0f - mask);

        // muled_pixel       = muled_pixel * wb_mul;

        // pixel             = muled_pixel;

        // if (pixel > maximum[color_idx]) {
        //   pixel = maximum[color_idx];
        // }

        // // pixel /= maximum[color_idx];

        img.at<float>(y, x) = pixel;
      }
    }
  }
}

};  // namespace CPU
};  // namespace puerhlab