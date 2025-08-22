#pragma once

#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"

#include <libraw/libraw_types.h>
#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"

namespace puerhlab {
namespace CPU {
inline static auto GetBlackSom(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  const auto           base_black_level = static_cast<float>(raw_data.color.black);
  std::array<float, 3> black_level      = {
      (base_black_level + static_cast<float>(raw_data.color.cblack[0])) / 65535.0f,
      (base_black_level + static_cast<float>(raw_data.color.cblack[1])) / 65535.0f,
      (base_black_level + static_cast<float>(raw_data.color.cblack[2])) / 65535.0f};

  return black_level;
}

inline static auto GetScaleMul(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  auto                 pre_mul     = raw_data.color.pre_mul;
  float                max_pre_mul = std::max({pre_mul[0], pre_mul[1], pre_mul[2]});

  auto                 c_white     = (int)raw_data.color.maximum;
  auto                 c_black     = (int)raw_data.color.black;

  std::array<float, 3> scale_mul;
  for (int c = 0; c < 3; ++c) {
    scale_mul[c] = (pre_mul[c] / max_pre_mul) / ((c_white - c_black) / 65535.0f);
  }

  return scale_mul;
}

inline static auto GetClMax(const libraw_rawdata_t& raw_data) -> std::array<float, 3> {
  std::array<float, 3> cl_max;

  int                  c_white    = (int)raw_data.color.maximum;

  std::array<float, 3> cblack_som = GetBlackSom(raw_data);
  std::array<float, 3> scale_mul  = GetScaleMul(raw_data);

  for (int c = 0; c < 3; ++c) {
    cl_max[c] = (c_white = cblack_som[c]) * scale_mul[c];
  }

  return cl_max;
}

inline static auto GetChMax(cv::Mat& img) -> std::array<float, 3> {
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(img.cols / 16, img.rows / 16));

  std::array<float, 3> ch_max = {0, 0, 0};

  resized.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    if (pixel[0] > ch_max[0]) {
      ch_max[0] = pixel[0];
    }
    if (pixel[1] > ch_max[1]) {
      ch_max[1] = pixel[1];
    }
    if (pixel[2] > ch_max[2]) {
      ch_max[2] = pixel[2];
    }
  });

  return ch_max;
}
/**
 * @brief Adapted from CarVac/librtprocess/src/postprocess/hilite_recon.cc
 *
 * @param img
 * @param raw_processor
 */
void HighlightReconstruct(cv::Mat& img, LibRaw& raw_processor) {
  const int       width       = img.cols;
  const int       height      = img.rows;

  constexpr int   range       = 2;
  constexpr int   pitch       = 4;

  constexpr float threshpct   = 0.25f;
  constexpr float maxpct      = 0.95f;
  constexpr float epsilon     = 1e-5f;

  constexpr float blendthresh = 1.0;
  constexpr int   ColorCount  = 3;

  cv::Mat1f       red;
  cv::Mat1f       green;
  cv::Mat1f       blue;

  cv::extractChannel(img, red, 0);
  cv::extractChannel(img, red, 1);
  cv::extractChannel(img, red, 2);
  auto            black_lev                     = raw_processor.imgdata.color.black;
  auto            c_black                       = GetBlackSom(raw_processor.imgdata.rawdata);

  constexpr float trans[ColorCount][ColorCount] = {
      {1.f, 1.f, 1.f}, {1.7320508f, -1.7320508f, 0.f}, {-1.f, -1.f, 2.f}};
  constexpr float itrans[ColorCount][ColorCount] = {
      {1.f, 0.8660254f, -0.5f}, {1.f, -0.8660254f, -0.5f}, {1.f, 0.f, 1.f}};

  float factor[3];
  auto  ch_max = GetChMax(img);
  auto  cl_max = GetClMax(raw_processor.imgdata.rawdata);

  for (int c = 0; c < ColorCount; ++c) {
    factor[c] = ch_max[c] / cl_max[c];
  }

  float min_factor = std::min({factor[0], factor[1], factor[2]});

  if (min_factor > 1.f) {  // all three channels clipped
    // calculate clip factor per channel
    for (int c = 0; c < ColorCount; ++c) {
      factor[c] /= min_factor;
    }
    // Get max clip factor
    int   max_pos     = 0;
    float max_val_new = 0.f;

    for (int c = 0; c < ColorCount; ++c) {
      if (ch_max[c] / factor[c] > max_val_new) {
        max_val_new = ch_max[c] / factor[c];
        max_pos     = c;
      }
    }

    float clip_factor = cl_max[max_pos] / max_val_new;

    if (clip_factor < maxpct) {
      // if max clipFactor < maxpct (0.95) adjust per channel factors
      for (int c = 0; c < ColorCount; ++c) {
        factor[c] *= (maxpct / clip_factor);
      }
    }
  } else {
    factor[0] = factor[1] = factor[2] = 1.f;
  }

  float max_f[3], thresh[3];

  for (int c = 0; c < ColorCount; ++c) {
    thresh[c] = ch_max[c] * threshpct / factor[c];
    max_f[c]  = ch_max[c] * maxpct / factor[c];
  }

  float white_pt = std::max({max_f[0], max_f[1], max_f[2]});
  float clip_pt  = std::min({max_f[0], max_f[1], max_f[2]});

  float med_pt   = max_f[0] + max_f[1] + max_f[2] - white_pt - clip_pt;

  float blend_pt = blendthresh + clip_pt;
  float med_factor[3];

  for (int c; c < ColorCount; ++c) {
    med_factor[c] = std::max(1.0f, (max_f[c] / med_pt) / (-blend_pt));
  }

  int min_x = width - 1;
  int max_x = 0;
  int min_y = height - 1;
  int max_y = 0;

#pragma omp parallel for schedule(dynamic, 16)
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (red(y, x) >= max_f[0] || green(y, x) >= max_f[1] || blue(y, x) >= max_f[2]) {
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
      }
    }
  }

  constexpr int blur_border            = 256;
  min_x                                = std::max(0, min_x - blur_border);
  min_y                                = std::max(0, min_y - blur_border);
  max_x                                = std::min(width - 1, max_x + blur_border);
  max_y                                = std::min(height - 1, max_y + blur_border);

  const int                blur_width  = max_x - min_x + 1;
  const int                blur_height = max_y - min_y + 1;

  std::array<cv::Mat1f, 3> channel_blur;
  for (int c = 0; c < ColorCount; ++c) {
    channel_blur[c].create(blur_height, blur_width);
  }

  cv::Mat1f temp(blur_height, blur_width);

  boxblur2(red, channel_blur[0], temp, min_y, min_x, blur_height, blur_width, 4);
  boxblur2(green, channel_blur[1], temp, min_y, min_x, blur_height, blur_width, 4);
  boxblur2(blue, channel_blur[2], temp, min_y, min_x, blur_height, blur_width, 4);

#pragma omp parallel for
  for (int y = 0; y < blur_height; ++y) {
    for (int x = 0; x < blur_width; ++x) {
      channel_blur[0](y, x) = fabsf(channel_blur[0](y, x) - red(y, x)) +
                              fabsf(channel_blur[1](y, x) - green(y, x)) +
                              fabsf(channel_blur[2](y, x) - blue(y, x));
    }
  }

  for (int c = 1; c < 3; ++c) {
    // Free up some memory
    channel_blur[c].release();
  }

  std::array<cv::Mat1f, 4> hilite_full;
  for (cv::Mat1f& channel : hilite_full) {
    channel.create(blur_height, blur_width);
    channel.setTo(0.0f);
  }

  double highpass_sum  = 0.f;
  int    highpass_norm = 0;

#pragma omp parallel for schedule(dynamic, 16)
  for (int y = 0; y < blur_height; ++y) {
    for (int x = 0; x < blur_width; ++x) {
      if ((red(y + min_y, x + min_x) > thresh[0] || green(y + min_y, x + min_x) > thresh[1] ||
           blue(y + min_y, x + min_x) > thresh[2]) &&
          (red(y + min_y, x + min_x) > max_f[0] || green(y + min_y, x + min_x) > max_f[1] ||
           blue(y + min_y, x + min_x) > max_f[2])) {
        highpass_sum += static_cast<double>(channel_blur[0](y, x));
        ++highpass_norm;

        hilite_full[0](y, x) = red(y + min_y, x + min_x);
        hilite_full[1](y, x) = green(y + min_y, x + min_x);
        hilite_full[2](y, x) = blue(y + min_y, x + min_x);
        hilite_full[3](y, x) = 1.f;
      }
    }
  }

  float     highpass_avg = 2.0 * highpass_sum / (highpass_norm + static_cast<double>(epsilon));

  cv::Mat1f hilite_full4(blur_height, blur_width);
  // Blue highlight data
  boxblur2(hilite_full[3], hilite_full4, temp, 0, 0, blur_height, blur_width, 1);

  temp.release();

#pragma omp parallel for schedule(dynamic, 16)
  for (int y = 0; y < blur_height; ++y) {
    for (int x = 0; x < blur_width; ++x) {
      if (channel_blur[0](y, x) > highpass_avg) {
        // too much variation
        hilite_full[0](y, x) = hilite_full[1](y, x) = hilite_full[2](y, x) = hilite_full[3](y, x) =
            0.f;
        continue;
      }

      if (hilite_full4(y, x) > epsilon && hilite_full4(y, x) < 0.95f) {
        // too near an edge, could risk using CA affected pixels, therefore omit
        hilite_full[0](y, x) = hilite_full[1](y, x) = hilite_full[2](y, x) = hilite_full[3](y, x) =
            0.f;
      }
    }
  }

  channel_blur[0].release();
  hilite_full4.release();

  int                      hfh = (blur_height - (blur_height % pitch)) / pitch;
  int                      hfw = (blur_width - (blur_width % pitch)) / pitch;

  std::array<cv::Mat1f, 4> hilite;
  for (cv::Mat1f& channel : hilite) {
    channel.create(hfh + 1, hfw + 1);
    channel.setTo(0.0f);
  }

  cv::Mat1f temp2(blur_height, (blur_width / pitch) + ((blur_width % pitch) == 0 ? 0 : 1));
  for (int m = 0; m < 4; ++m) {
    boxblur_resamp(hilite_full[m], hilite[m], temp2, blur_height, blur_width, range, pitch);
  }

  temp2.release();

  std::array<cv::Mat1f, 8> hilite_dir;
  for (cv::Mat1f& channel : hilite_dir) {
    channel.create(hfh, hfw);
    channel.setTo(0.0f);
  }

  std::array<cv::Mat1f, 4> hilite_dir0;
  for (cv::Mat1f& channel : hilite_dir) {
    channel.create(hfw, hfh);
    channel.setTo(0.0f);
  }

  std::array<cv::Mat1f, 4> hilite_dir4;
  for (cv::Mat1f& channel : hilite_dir4) {
    channel.create(hfw, hfh);
    channel.setTo(0.0f);
  }

  // Fill gaps in highlight map by directional extension
  // Raster scan from four corners
}

};  // namespace CPU
};  // namespace puerhlab