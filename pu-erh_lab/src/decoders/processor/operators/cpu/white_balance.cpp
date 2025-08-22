#include "decoders/processor/operators/cpu/white_balance.hpp"

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

void WhiteBalanceCorrection(cv::Mat& img, LibRaw& raw_processor) {
  auto black_level = CalculateBlackLevel(raw_processor.imgdata.rawdata);
  auto wb          = GetWBCoeff(raw_processor.imgdata.rawdata);
  img.convertTo(img, CV_32FC1, 1.0f / 65535.0f);
  int w = img.cols;
  int h = img.rows;

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
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        int   color_idx   = raw_processor.COLOR(y, x);

        float pixel       = img.at<float>(y, x);

        pixel             = std::max(0.0f, pixel - black_level[color_idx]);

        float muled_pixel = pixel;
        float mask        = (color_idx == 0 || color_idx == 2) ? 1.0f : 0.0f;
        //
        float wb_mul      = (wb[color_idx] / wb[1]) * mask + (1.0f - mask);

        muled_pixel       = muled_pixel * wb_mul;

        pixel             = muled_pixel;

        if (pixel > maximum[color_idx]) {
          pixel = maximum[color_idx];
        }

        pixel /= maximum[color_idx];

        img.at<float>(y, x) = pixel;
      }
    }
  }
}

};  // namespace CPU
};  // namespace puerhlab