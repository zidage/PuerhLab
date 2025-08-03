#include "edit/operators/basic/exposure_op.hpp"

#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 */
ExposureOp::ExposureOp() : _exposure_offset(0.0f) { _scale = 1.0f; }

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 * @param exposure_offset
 */
ExposureOp::ExposureOp(float exposure_offset) : _exposure_offset(exposure_offset) {
  _scale = std::pow(2.0f, _exposure_offset);
}

ExposureOp::ExposureOp(const nlohmann::json& params) { SetParams(params); }

auto ExposureOp::Apply(ImageBuffer& input) -> ImageBuffer {
  cv::Mat& linear_image = input.GetCPUData();
  linear_image *= _scale;

  cv::min(linear_image, 1.0f, linear_image);
  cv::max(linear_image, 0.0f, linear_image);
  return {std::move(linear_image)};
}

auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
  _scale           = std::pow(2.0f, _exposure_offset);
}

};  // namespace puerhlab