#pragma once

#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

template <typename Derived>
class ToneRegionOp {
 private:
  auto ComputeOutput(float luminance, float adj) const -> float {
    return Derived::GetOutput(luminance, adj);
  }

 public:
  auto Apply(ImageBuffer& input) -> ImageBuffer {
    float    scale = static_cast<Derived*>(this)->GetScale();

    cv::Mat& img   = input.GetCPUData();
    if (img.depth() != CV_32F) {
      throw std::runtime_error("Tone region operator: Unsupported image format");
    }

    img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
      float Y        = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
      float Y_target = ComputeOutput(Y, scale);
      float delta    = Y_target - Y;
      for (int c = 0; c < 3; ++c) {
        float max_push = 1.0f - pixel[c];
        float min_push = -pixel[c];
        float push     = std::clamp(delta, min_push, max_push);
        pixel[c] += push;
      }
    });

    return {std::move(img)};
  }
};
}  // namespace puerhlab