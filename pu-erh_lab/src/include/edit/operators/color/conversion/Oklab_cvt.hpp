//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

#include "edit/operators/op_kernel.hpp"

namespace OklabCvt {
/**
 * @brief A 3-channel vector to represent an Oklab color
 *
 */
struct Oklab {
  float l_, a_, b_;
};

Oklab     LinearRGB2Oklab(const cv::Vec3f& rgb);

cv::Vec3f Oklab2LinearRGB(const Oklab& lab);

Oklab     ACESRGB2Oklab(const cv::Vec3f& rgb);
cv::Vec3f Oklab2ACESRGB(const Oklab& lab);

Oklab     ACESRGB2Oklab(const puerhlab::Pixel& pixel);
void      Oklab2ACESRGB(const Oklab& lab, puerhlab::Pixel& pixel);
};  // namespace OklabCvt
