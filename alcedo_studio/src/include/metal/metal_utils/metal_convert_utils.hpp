//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_METAL

#include <limits>
#include <opencv2/core/types.hpp>

namespace alcedo::metal {
class MetalImage;
}

namespace alcedo::metal::utils {

void ConvertTexture(const MetalImage& src, MetalImage& dst, double alpha = 1.0, double beta = 0.0);
void CropTexture(const MetalImage& src, MetalImage& dst, const cv::Rect& crop_rect);
void ClampTexture(MetalImage& image, float lo = std::numeric_limits<float>::lowest(),
                  float hi = 1.0f);
void Rotate180(MetalImage& image);
void Rotate90CW(MetalImage& image);
void Rotate90CCW(MetalImage& image);

}  // namespace alcedo::metal::utils

#endif
