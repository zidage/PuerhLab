#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {}

ShadowsOp::ShadowsOp(const nlohmann::json& params) { SetParams(params); }

void ShadowsOp::GetMask(cv::Mat& src, cv::Mat& mask) {
  static cv::Mat cached_mask;
  if (!cached_mask.empty()) {
    mask = cached_mask;
    return;
  }

  float   percentile = 66.0f;
  float   transition = 5.0f;

  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

  cv::Mat            flat = gray.reshape(1, 1);
  std::vector<float> gray_vector;
  flat.copyTo(gray_vector);
  std::sort(gray_vector.begin(), gray_vector.end());

  int   shadow_index     = static_cast<int>((gray_vector.size() - 1) * percentile / 100.0f);
  float shadow_threshold = gray_vector[shadow_index];

  float s_end            = std::min(shadow_threshold + transition, 1.0f);
  float s_start          = 0.0f;

  float range            = s_end - s_start;

  if (range <= 1e-6) {
    cv::threshold(gray, cached_mask, s_end, 1.0, cv::THRESH_BINARY_INV);
  } else {
    cv::subtract(gray, s_start, cached_mask);
    cached_mask = cached_mask / range;

    cv::subtract(1.0, cached_mask, cached_mask);
    cv::threshold(cached_mask, cached_mask, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(cached_mask, cached_mask, 1.0f, 1.0f, cv::THRESH_TRUNC);
  }
  cv::GaussianBlur(cached_mask, cached_mask, cv::Size(1, 1), 0);

  cv::cvtColor(cached_mask, cached_mask, cv::COLOR_GRAY2RGB);
  mask = cached_mask;
}

auto ShadowsOp::GetOutput(float luminance, float adj) -> float {
  float x = luminance;
  return x + adj * x * std::pow(1 - x, 2.0f) * std::exp(-40 * std::pow(x - 0.2f, 2.0f));
}

auto ShadowsOp::GetScale() -> float { return _offset / 100.0f; }

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