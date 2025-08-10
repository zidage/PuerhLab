#include "edit/operators/basic/highlight_op.hpp"

#include <easy/profiler.h>

#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
HighlightsOp::HighlightsOp(float offset) : _offset(offset) {}

HighlightsOp::HighlightsOp(const nlohmann::json& params) { SetParams(params); }

void HighlightsOp::GetMask(cv::Mat& src, cv::Mat& mask) {
  EASY_BLOCK("Get Shadow Mask");
  cv::Mat cached_mask;

  // Parameters
  float   percentile = 88.0f;  // shadows percentile
  float   transition = 60.0f;  // smooth zone width

  // Step 1: Downsample L channel before processing
  cv::Mat small_src;
  int     mask_scale = 8;  // 1/32 resolution
  cv::resize(src, small_src, cv::Size(src.cols / mask_scale, src.rows / mask_scale), 0, 0,
             cv::INTER_AREA);

  // Step 2: Histogram in [0, 100]
  int          histSize  = 128;
  float        range[]   = {0.0f, 100.0f};
  const float* histRange = {range};
  cv::Mat      hist;
  cv::calcHist(&small_src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  hist /= small_src.total();

  // Step 3: CDF
  std::vector<float> cdf(histSize);
  cdf[0] = hist.at<float>(0);
  for (int i = 1; i < histSize; ++i) cdf[i] = cdf[i - 1] + hist.at<float>(i);

  // Step 4: Threshold
  float threshold = 0.0f;
  for (int i = 0; i < histSize; ++i) {
    if (cdf[i] >= percentile / 100.0f) {
      threshold = range[0] + (range[1] - range[0]) * (i / static_cast<float>(histSize - 1));
      break;
    }
  }

  // Step 5: Low-res mask generation
  cv::Mat small_mask;
  if (transition <= 0.0f) {
    cv::threshold(small_src, small_mask, threshold, 1.0f, cv::THRESH_BINARY_INV);
  } else {
    cv::threshold(small_src, small_mask, threshold, 0.0f, cv::THRESH_TOZERO);  // L - threshold
    // cv::Mat resized;
    // cv::resize(small_mask, resized, cv::Size(512, 512));
    cv::imshow("Highlight Mask", small_mask);
    cv::waitKey(0);
    // small_mask /= transition;  // normalize

    // small_mask *= 100.0f;

    cv::threshold(small_mask, small_mask, 100.0f, 100.0f, cv::THRESH_TRUNC);
    cv::threshold(small_mask, small_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);
  }
  cv::GaussianBlur(small_mask, small_mask, cv::Size(13, 13), 0);

  // Step 6: Upsample back to original size
  cv::resize(small_mask, cached_mask, src.size(), 0, 0, cv::INTER_CUBIC);

  mask = cached_mask;

  EASY_END_BLOCK;
}

auto HighlightsOp::GetOutput(float luminance, float adj) -> float {
  if (luminance < 40.0f) {
    return 0.0f;  // No highlights below 40
  }
  return SmoothStep(40.0f, 100.0f, luminance) * adj;
}

auto HighlightsOp::GetScale() -> float { return _offset / 300.0f; }

auto HighlightsOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<HighlightsOp>::Apply(input);
}

auto HighlightsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void HighlightsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab