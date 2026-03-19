//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <libraw/libraw.h>

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <string>

namespace puerhlab {

#if defined(__CUDACC__)
#define PUERHLAB_HD __host__ __device__
#else
#define PUERHLAB_HD
#endif

enum class RawInputKind {
  BayerRaw,
  DebayeredRgb,
  Unsupported,
};

struct BayerPattern2x2 {
  int raw_fc[4] = {0, 1, 3, 2};
  int rgb_fc[4] = {0, 1, 1, 2};
};

PUERHLAB_HD inline auto BayerCellIndex(const int y, const int x) -> int {
  return ((y & 1) << 1) | (x & 1);
}

PUERHLAB_HD inline auto FoldRawColorToRgb(const int raw_color) -> int {
  switch (raw_color) {
    case 0:
      return 0;
    case 2:
      return 2;
    case 1:
    case 3:
      return 1;
    default:
      return raw_color;
  }
}

PUERHLAB_HD inline auto RawColorAt(const BayerPattern2x2& pattern, const int y, const int x) -> int {
  return pattern.raw_fc[BayerCellIndex(y, x)];
}

PUERHLAB_HD inline auto RgbColorAt(const BayerPattern2x2& pattern, const int y, const int x) -> int {
  return pattern.rgb_fc[BayerCellIndex(y, x)];
}

inline auto DescribeRgbColor(const int color) -> char {
  switch (color) {
    case 0:
      return 'R';
    case 1:
      return 'G';
    case 2:
      return 'B';
    default:
      return '?';
  }
}

inline auto DescribeBayerPattern(const BayerPattern2x2& pattern) -> std::string {
  std::string desc(4, '?');
  for (int i = 0; i < 4; ++i) {
    desc[i] = DescribeRgbColor(pattern.rgb_fc[i]);
  }
  return desc;
}

inline auto IsClassic2x2Bayer(const BayerPattern2x2& pattern) -> bool {
  int raw_sorted[4] = {pattern.raw_fc[0], pattern.raw_fc[1], pattern.raw_fc[2], pattern.raw_fc[3]};
  std::sort(std::begin(raw_sorted), std::end(raw_sorted));

  int rgb_sorted[4] = {pattern.rgb_fc[0], pattern.rgb_fc[1], pattern.rgb_fc[2], pattern.rgb_fc[3]};
  std::sort(std::begin(rgb_sorted), std::end(rgb_sorted));

  static constexpr int kExpectedRaw[4] = {0, 1, 2, 3};
  static constexpr int kExpectedRgb[4] = {0, 1, 1, 2};

  return std::equal(std::begin(raw_sorted), std::end(raw_sorted), std::begin(kExpectedRaw)) &&
         std::equal(std::begin(rgb_sorted), std::end(rgb_sorted), std::begin(kExpectedRgb));
}

inline auto IsRGGBPattern(const BayerPattern2x2& pattern) -> bool {
  static constexpr int kRggbRaw[4] = {0, 1, 3, 2};
  return std::equal(std::begin(pattern.raw_fc), std::end(pattern.raw_fc), std::begin(kRggbRaw));
}

inline auto ReadLibRawBayerPattern(LibRaw& raw_processor) -> BayerPattern2x2 {
  BayerPattern2x2 pattern = {};
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      const int index       = BayerCellIndex(row, col);
      pattern.raw_fc[index] = raw_processor.COLOR(row, col);
      pattern.rgb_fc[index] = FoldRawColorToRgb(pattern.raw_fc[index]);
    }
  }
  return pattern;
}

inline auto ClassifyRawInput(const libraw_rawdata_t& raw_data) -> RawInputKind {
  if (raw_data.raw_image != nullptr) {
    return RawInputKind::BayerRaw;
  }
  if (raw_data.color3_image != nullptr || raw_data.float3_image != nullptr) {
    return RawInputKind::DebayeredRgb;
  }
  return RawInputKind::Unsupported;
}

template <typename T>
auto DownsampleBayer2xTyped(const cv::Mat& src, const BayerPattern2x2& pattern) -> cv::Mat {
  const int out_rows = src.rows / 2;
  const int out_cols = src.cols / 2;
  cv::Mat   dst(out_rows, out_cols, src.type());

  for (int y = 0; y < out_rows; ++y) {
    T* drow = dst.ptr<T>(y);
    for (int x = 0; x < out_cols; ++x) {
      const int dst_index       = BayerCellIndex(y, x);
      const int expected_color  = pattern.raw_fc[dst_index];
      int       src_index       = -1;
      for (int i = 0; i < 4; ++i) {
        if (pattern.raw_fc[i] == expected_color) {
          src_index = i;
          break;
        }
      }
      if (src_index < 0) {
        throw std::runtime_error("RawProcessor: invalid Bayer pattern for downsample");
      }

      const int src_row = 2 * y + (src_index >> 1);
      const int src_col = 2 * x + (src_index & 1);
      drow[x]           = src.ptr<T>(src_row)[src_col];
    }
  }

  return dst;
}

inline auto DownsampleBayer2x(const cv::Mat& src, const BayerPattern2x2& pattern) -> cv::Mat {
  switch (src.type()) {
    case CV_32FC1:
      return DownsampleBayer2xTyped<float>(src, pattern);
    case CV_16UC1:
      return DownsampleBayer2xTyped<uint16_t>(src, pattern);
    default:
      throw std::runtime_error("RawProcessor: unsupported Bayer type for downsample");
  }
}

}  // namespace puerhlab

#undef PUERHLAB_HD
