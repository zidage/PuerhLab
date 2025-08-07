#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {}

ShadowsOp::ShadowsOp(const nlohmann::json& params) { SetParams(params); }

void ShadowsOp::GetMask(cv::Mat& src, cv::Mat& mask) {
  EASY_BLOCK("Get Shadow Mask");
  static cv::Mat cached_mask;
  if (!cached_mask.empty()) {
    mask = cached_mask;
    return;
  }

  // For shadows, we look at the lower end of the brightness distribution.
  // e.g., the darkest 23% of pixels. You can adjust this value.
  float        percentile = 65.0f;
  float        transition = 0.0f;  // A non-zero transition creates a smoother gradient

  // Histogram and CDF calculation (remains the same)
  int          histSize   = 512;
  float        range[]    = {0.0f, 100.0f};
  const float* histRange  = {range};
  cv::Mat      hist;
  cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  hist /= src.total();

  std::vector<float> cdf(histSize, 0.0f);
  cdf[0] = hist.at<float>(0);
  for (int i = 1; i < histSize; ++i) {
    cdf[i] = cdf[i - 1] + hist.at<float>(i);
  }

  // Find the brightness threshold for the specified percentile
  float threshold = 0.0f;  // Default to 0
  for (int i = 0; i < histSize; ++i) {
    if (cdf[i] >= percentile / 100.0f) {
      threshold = range[0] + (range[1] - range[0]) * (i / static_cast<float>(histSize));
      break;
    }
  }

  // --- Mask Generation Logic (Modified for Shadows) ---

  // Define the range for shadows: from 0 up to the threshold + transition.
  float s_start = 0.0f;
  float s_end   = std::min(threshold + transition, 100.0f);
  float range_v = s_end - s_start;

  if (range_v <= 1e-6) {
    // If the range is negligible, create a binary mask.
    // For shadows, pixels <= threshold are shadows (value 100).
    cv::threshold(src, cached_mask, s_end, 100.0f, cv::THRESH_BINARY_INV);
  } else {
    // Create a gradient mask for the shadow range.
    // The mask value should be 100 at src=0 and 0 at src=s_end.
    // Formula: mask = (s_end - src) * 100 / range_v

    // 1. Calculate (s_end - src)
    cv::subtract(cv::Scalar(s_end), src, cached_mask);

    // 2. Scale the result to the [0, 100] range
    // Note: The original code's division led to a scaling issue.
    // Multiplying ensures a full [0, 100] gradient.
    cached_mask = cached_mask * 100.0f;
    // 3. Clamp values to the [0, 100] range
    cv::threshold(cached_mask, cached_mask, 100.0f, 100.0f,
                  cv::THRESH_TRUNC);                                         // values > 100 => 100
    cv::threshold(cached_mask, cached_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);  // values < 0   => 0
  }

  // // Apply a small blur for smoothness (optional)
  cv::GaussianBlur(cached_mask, cached_mask, cv::Size(101, 101), 0);

  // cv::cvtColor(cached_mask, cached_mask, cv::COLOR_GRAY2RGB);
  // cv::Mat resized;
  // cv::resize(cached_mask, resized, cv::Size(512, 512));
  // cv::imshow("Shadow Mask", resized);
  // cv::waitKey(0);
  mask = cached_mask;
  EASY_END_BLOCK;
}

auto ShadowsOp::GetOutput(float luminance, float adj) -> float {
  float x = luminance;
  return x + adj * x * std::pow(1 - x, 2.0f) * std::exp(-40 * std::pow(x - 0.2f, 2.0f));
}

auto ShadowsOp::GetScale() -> float { return _offset / 500.0f; }

auto ShadowsOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<ShadowsOp>::Apply(input);
}

auto ShadowsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab