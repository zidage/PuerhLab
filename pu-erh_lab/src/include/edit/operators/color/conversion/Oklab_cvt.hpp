#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

namespace OklabCvt {
/**
 * @brief A 3-channel vector to represent an Oklab color
 *
 */
struct Oklab {
  float L, a, b;
};

Oklab     LinearRGB2Oklab(const cv::Vec3f& rgb);

cv::Vec3f Oklab2LinearRGB(const Oklab& lab);
};  // namespace OklabCvt
