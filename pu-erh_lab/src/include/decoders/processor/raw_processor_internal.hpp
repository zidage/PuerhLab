//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include <libraw/libraw.h>
#include <opencv2/core.hpp>

#include "decoders/processor/raw_processor.hpp"

namespace puerhlab::detail {

enum class CudaExecutionMode {
  FullFrame,
  Tiled,
};

inline constexpr int kCudaTileThresholdLongEdge = 8500;
inline constexpr int kCudaTileInnerSize         = 4096;
inline constexpr int kCudaTileHaloSize          = 16;

inline auto DecodeResScaleDivisor(const DecodeRes decode_res) -> int {
  switch (decode_res) {
    case DecodeRes::FULL:
      return 1;
    case DecodeRes::HALF:
      return 2;
    case DecodeRes::QUARTER:
      return 4;
    case DecodeRes::EIGHTH:
      return 8;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }
}

inline auto ScaleCoordFloor(const int value, const int divisor) -> int { return value / divisor; }

inline auto ScaleCoordCeil(const int value, const int divisor) -> int {
  return (value + divisor - 1) / divisor;
}

inline auto BuildActiveAreaRect(const libraw_image_sizes_t& sizes, const cv::Size& image_size,
                                const int scale_divisor = 1) -> cv::Rect {
  const int raw_width  = std::max(static_cast<int>(sizes.raw_width), 0);
  const int raw_height = std::max(static_cast<int>(sizes.raw_height), 0);

  const int raw_left   = std::clamp(static_cast<int>(sizes.left_margin), 0, raw_width);
  const int raw_top    = std::clamp(static_cast<int>(sizes.top_margin), 0, raw_height);
  const int raw_right  = std::clamp(raw_left + static_cast<int>(sizes.width), raw_left, raw_width);
  const int raw_bottom = std::clamp(raw_top + static_cast<int>(sizes.height), raw_top, raw_height);

  const int left       = std::clamp(ScaleCoordFloor(raw_left, scale_divisor), 0, image_size.width);
  const int top        = std::clamp(ScaleCoordFloor(raw_top, scale_divisor), 0, image_size.height);
  const int right  = std::clamp(ScaleCoordCeil(raw_right, scale_divisor), left, image_size.width);
  const int bottom = std::clamp(ScaleCoordCeil(raw_bottom, scale_divisor), top, image_size.height);
  const int width  = right - left;
  const int height = bottom - top;

  if (width <= 0 || height <= 0) {
    return {0, 0, image_size.width, image_size.height};
  }
  return {left, top, width, height};
}

inline auto BuildDecodeCropRect(const libraw_image_sizes_t& sizes, const cv::Size& image_size,
                                const DecodeRes decode_res) -> cv::Rect {
  return BuildActiveAreaRect(sizes, image_size, DecodeResScaleDivisor(decode_res));
}

inline auto IsFullImageRect(const cv::Rect& rect, const cv::Size& image_size) -> bool {
  return rect.x == 0 && rect.y == 0 && rect.width == image_size.width &&
         rect.height == image_size.height;
}

auto SelectCudaExecutionMode(const RawParams& params, const RawCfaPattern& cfa_pattern,
                             const cv::Rect& active_rect) -> CudaExecutionMode;

void SetCudaExecutionModeOverrideForTesting(const std::optional<CudaExecutionMode>& mode);
auto GetCudaExecutionModeOverrideForTesting() -> std::optional<CudaExecutionMode>;

}  // namespace puerhlab::detail
