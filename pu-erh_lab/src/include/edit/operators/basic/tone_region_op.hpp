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
    float* img_data         = reinterpret_cast<float*>(img.data);
    int    total_floats_img = static_cast<int>(img.total() * img.channels());

    if constexpr (Derived::_tone_region == ToneRegion::SHADOWS) {
      cv::Mat mask;
      Derived::GetMask(img, mask);

      const hw::ScalableTag<float> d;
      int                          lanes = static_cast<int>(hw::Lanes(d));
      CV_Assert(mask.isContinuous());

      float* mask_data = reinterpret_cast<float*>(mask.data);

      cv::parallel_for_(
          cv::Range(0, total_floats_img),
          [&](const cv::Range& range) {
            int    i           = range.start;
            int    end         = range.end;

            auto   v_scale     = hw::Set(d, scale);

            size_t aligned_end = i + ((end - i) / lanes) * lanes;
            for (; i < aligned_end; i += lanes) {
              auto v_img  = hw::Load(d, img_data + i);
              auto v_mask = hw::Load(d, mask_data + i);
              v_img       = hw::MulAdd(v_mask, v_scale, v_img);
              hw::Store(v_img, d, img_data + i);
            }

            for (; i < end; ++i) {
              float mask_data_remains = mask_data[i];
              img_data[i]             = mask_data_remains * scale + img_data[i];
            }
          },
          cv::getNumThreads() * 4);
    } else if constexpr (Derived::_tone_region == ToneRegion::HIGHLIGHTS) {
      img.forEach<float>([&](float& pixel, const int*) {
        float offset = Derived::GetOutput(pixel, scale);
        pixel        = pixel + offset;
      });
    } else {
      const hw::ScalableTag<float> d;
      int                          lanes = static_cast<int>(hw::Lanes(d));
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
              img_data[i] = static_cast<Derived*>(this)->GetOutput(img_data[i], scale);
            }
          },
          cv::getNumThreads() * 4);
    }
    EASY_END_BLOCK;
    return {std::move(img)};
  }
};
}  // namespace puerhlab