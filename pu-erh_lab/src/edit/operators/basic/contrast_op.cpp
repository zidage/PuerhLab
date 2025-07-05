#include "edit/operators/basic/contrast_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <stdexcept>

namespace puerhlab {
ContrastOp::ContrastOp() : _contrast_offset(0.0f) { _scale = 1.0f; }

ContrastOp::ContrastOp(float contrast_offset) : _contrast_offset(contrast_offset) {
  _scale = std::exp(contrast_offset / 100.0f);
}

auto ContrastOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& linear_image = input.GetCPUData();

  if (linear_image.depth() != CV_32F) {
    throw std::runtime_error("Contrast operator: Unsupported image format");
  }

  linear_image = (linear_image - 0.5f) * _scale + 0.5f;
  cv::min(linear_image, 1.0f, linear_image);
  cv::max(linear_image, 0.0f, linear_image);

  return {std::move(linear_image)};
}

auto ContrastOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _contrast_offset;
  return o;
}

void ContrastOp::SetParams(const nlohmann::json& params) {
  _contrast_offset = params[GetScriptName()];
  _scale           = std::exp(_contrast_offset / 100.0f);
}
};  // namespace puerhlab