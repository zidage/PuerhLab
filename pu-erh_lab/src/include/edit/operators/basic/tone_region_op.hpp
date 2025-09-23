#pragma once

#include <hwy/highway.h>

#include <algorithm>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/functions.hpp"

namespace hw = hwy::HWY_NAMESPACE;
namespace puerhlab {
struct LinearToneCurve {
  hw::Vec<hw::ScalableTag<float>> white_point;
  hw::Vec<hw::ScalableTag<float>> black_point;
  hw::Vec<hw::ScalableTag<float>> slope;
};

enum class ToneRegion {
  HIGHLIGHTS,
  SHADOWS,
  BLACK,
  WHITE,
};
template <typename Derived>

class ToneRegionOp {
 private:
  auto ComputeOutput(float luminance, float adj) const -> float {
    return Derived::GetOutput(luminance, adj);
  }

 public:
  void Apply(std::shared_ptr<ImageBuffer> input) {
    cv::Mat& img = input->GetCPUData();
    if (img.depth() != CV_32F) {
      throw std::runtime_error("Tone region operator: Unsupported image format");
    }
    CV_Assert(img.isContinuous());
    float*                       img_data         = reinterpret_cast<float*>(img.data);
    int                          total_floats_img = static_cast<int>(img.total() * img.channels());
    const hw::ScalableTag<float> d;
    int                          lanes   = static_cast<int>(hw::Lanes(d));

    auto*                        derived = static_cast<Derived*>(this);
    // For all tone regions, we can directly apply the adjustment
    // using a tone curve.
    // if constexpr (Derived::_tone_region == ToneRegion::SHADOWS) {
    img.forEach<cv::Vec3f>([this](cv::Vec3f& pixel, const int* /* position */) {
      pixel = static_cast<Derived*>(this)->GetOutput(pixel);
    });
  }
};
}  // namespace puerhlab