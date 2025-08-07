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
  EASY_BLOCK("Get Highlight Mask");
  static cv::Mat cached_mask;
  if (!cached_mask.empty()) {
    mask = cached_mask;
    return;
  }

  float        percentile = 77.0f;
  float        transition = 0.0f;

  // cv::Mat gray;
  // cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

  int          histSize   = 256;
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

  float threshold = 100.0f;
  for (int i = 0; i < histSize; ++i) {
    if (cdf[i] >= percentile / 100.0f) {
      threshold = range[0] + (range[1] - range[0]) * (i / static_cast<float>(histSize));
      break;
    }
  }

  float h_start = std::max(threshold - transition, 0.0f);
  float h_end   = 100.0f;
  float range_v = h_end - h_start;

  if (range_v <= 1e-6) {
    cv::threshold(src, cached_mask, h_start, 100.0f, cv::THRESH_BINARY);
  } else {
    cv::subtract(src, h_start, cached_mask);
    cached_mask = cached_mask / range_v;

    cv::subtract(100.0f, cached_mask, cached_mask);
    cached_mask *= 100.0f;
    cv::threshold(cached_mask, cached_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(cached_mask, cached_mask, 100.0f, 100.0f, cv::THRESH_TRUNC);
  }

  cv::GaussianBlur(cached_mask, cached_mask, cv::Size(1, 1), 0);
  // cached_mask = 100.0f - cached_mask;
  // cv::cvtColor(cached_mask, cached_mask, cv::COLOR_GRAY2RGB);
  mask = cached_mask;
  cv::Mat resized;
  cv::resize(cached_mask, resized, cv::Size(512, 512));
  cv::imshow("Shadow Mask", resized);
  cv::waitKey(0);

  EASY_END_BLOCK;
}

auto HighlightsOp::GetOutput(float luminance, float adj) -> float {
  if (luminance <= 0.4f) {
    return luminance;
  } else if (luminance > 0.4f && luminance <= 1.0f) {
    float term = 10.0f * luminance - 4.0f;
    return luminance + adj * luminance * std::pow(term, 1.1f) / 15.0f;
  } else {
    return 1.0f;
  }
}

auto HighlightsOp::GetScale() -> float { return _offset / 100.0f; }

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