#pragma once

#include <easy/profiler.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/hal/intrin_sse.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>

#include "edit/operators/op_base.hpp"
#include "edit/operators/utils/functions.hpp"

namespace puerhlab {
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

    if constexpr (Derived::_tone_region == ToneRegion::HIGHLIGHTS ||
                  Derived::_tone_region == ToneRegion::SHADOWS) {
      cv::Mat mask;
      Derived::GetMask(img, mask);

      CV_Assert(img.isContinuous() && mask.isContinuous());
      float* img_data         = reinterpret_cast<float*>(img.data);
      float* mask_data        = reinterpret_cast<float*>(mask.data);

      size_t total_floats_img = img.total() * img.channels();

      cv::parallel_for_(
          cv::Range(0, total_floats_img),
          [&](const cv::Range& range) {
            int             i           = range.start;
            int             end         = range.end;

            cv::v_float32x4 v_scale     = cv::v_setall_f32(scale);
            int             aligned_end = i + ((end - i) / 4) * 4;
            for (; i < aligned_end; i += 4) {
              cv::v_float32x4 v_img  = cv::v_load(img_data + i);
              cv::v_float32x4 v_mask = cv::v_load(mask_data + i);
              v_img                  = cv::v_muladd(v_mask, v_scale, v_img);
              cv::v_store(img_data + i, v_img);
            }

            for (; i < end; ++i) {
              float mask_data_remains = mask_data[i];
              img_data[i]             = mask_data_remains * scale + img_data[i];
            }
          },
          cv::getNumThreads() * 4);
      // img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int* pos) {
      //   float weight = mask.at<float>(pos[0], pos[1]);
      //   float delta  = weight * scale;
      //   cv::add(pixel, delta, pixel);
      //   cv::threshold(pixel, pixel, 0.0f, 0.0f, cv::THRESH_TOZERO);
      //   cv::threshold(pixel, pixel, 1.0f, 1.0f, cv::THRESH_TRUNC);
      // });
      // cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
      // cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
    } else {
      img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
        for (int c = 0; c < 3; ++c) {
          pixel[c] = ComputeOutput(pixel[c], scale);
        }
      });
    }
    EASY_END_BLOCK;
    return {std::move(img)};
  }
};
}  // namespace puerhlab