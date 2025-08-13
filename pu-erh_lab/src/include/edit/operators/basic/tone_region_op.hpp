#pragma once

#include <easy/profiler.h>
#include <hwy/highway.h>

#include <algorithm>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/intrin_sse.hpp>
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
  auto Apply(ImageBuffer& input) -> ImageBuffer {
    EASY_BLOCK("Tone Region Pipeline");
    float    scale = static_cast<Derived*>(this)->GetScale();

    cv::Mat& img   = input.GetCPUData();
    if (img.depth() != CV_32F) {
      throw std::runtime_error("Tone region operator: Unsupported image format");
    }
    CV_Assert(img.isContinuous());
    float*                       img_data         = reinterpret_cast<float*>(img.data);
    int                          total_floats_img = static_cast<int>(img.total() * img.channels());
    const hw::ScalableTag<float> d;
    int                          lanes = static_cast<int>(hw::Lanes(d));

    // For all tone regions, we can directly apply the adjustment
    // using a tone curve.
    cv::parallel_for_(
        cv::Range(0, total_floats_img),
        [&](const cv::Range& range) {
          int i           = range.start;
          int end         = range.end;

          int aligned_end = i + ((end - i) / lanes) * lanes;
          for (; i < aligned_end; i += lanes) {
            auto v_img = hw::Load(d, img_data + i);
            v_img      = static_cast<Derived*>(this)->GetOutput(v_img);
            hw::Store(v_img, d, img_data + i);
          }

          for (; i < end; ++i) {
            img_data[i] = static_cast<Derived*>(this)->GetOutput(img_data[i]);
          }
        },
        cv::getNumThreads() * 4);

    EASY_END_BLOCK;
    return {std::move(img)};
  }
};
}  // namespace puerhlab