//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
