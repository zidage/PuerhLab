#include "edit/operators/basic/highlight_op.hpp"

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
  static cv::Mat cached_mask;
  if (!cached_mask.empty()) {
    mask = cached_mask;
    return;
  }

  float   percentile = 77.0f;
  float   transition = 0.0f;

  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

  cv::Mat            flat = gray.reshape(1, 1);
  std::vector<float> gray_vector;
  flat.copyTo(gray_vector);
  std::sort(gray_vector.begin(), gray_vector.end());

  int   highlight_index = static_cast<int>((gray_vector.size() - 1) * percentile / 100.0f);
  float threshold       = gray_vector[highlight_index];

  float h_start         = std::max(threshold - transition, 0.0f);
  float h_end           = 1.0f;

  float range           = h_end - h_start;

  if (range <= 1e-6) {
    cv::threshold(gray, cached_mask, h_start, 1.0, cv::THRESH_BINARY);
  } else {
    cv::subtract(gray, h_start, cached_mask);
    cached_mask = cached_mask / range;

    cv::subtract(1.0, cached_mask, cached_mask);
    cv::threshold(cached_mask, cached_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(cached_mask, cached_mask, 1.0f, 1.0f, cv::THRESH_TRUNC);
  }
  cv::GaussianBlur(cached_mask, cached_mask, cv::Size(1, 1), 0);
  cached_mask = 1.0f - cached_mask;
  cv::cvtColor(cached_mask, cached_mask, cv::COLOR_GRAY2RGB);
  mask = cached_mask;
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