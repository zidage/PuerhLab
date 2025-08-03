#include "edit/operators/basic/contrast_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <stdexcept>

#include "edit/operators/operator_factory.hpp"

namespace puerhlab {

/**
 * @brief Default construct a new Contrast Op:: Contrast Op object
 *
 */
ContrastOp::ContrastOp() : _contrast_offset(0.0f) { _scale = 1.0f; }

/**
 * @brief Construct a new Contrast Op:: Contrast Op object
 *
 * @param contrast_offset
 */
ContrastOp::ContrastOp(float contrast_offset) : _contrast_offset(contrast_offset) {
  _scale = std::exp(contrast_offset / 100.0f);
}

ContrastOp::ContrastOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Apply the contrast adjustment
 *
 * @param input
 * @return ImageBuffer
 */
auto ContrastOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& linear_image = input.GetCPUData();

  if (linear_image.depth() != CV_32F) {
    throw std::runtime_error("Contrast operator: Unsupported image format");
  }

  // TODO: Change to CLAHE (Contrast Limited Adaptive Histogram Equalization)
  linear_image = (linear_image - 0.5f) * _scale + 0.5f;
  // clamp
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