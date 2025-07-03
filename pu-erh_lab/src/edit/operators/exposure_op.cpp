#include "edit/operators/exposure_op.hpp"

#include <opencv2/core/mat.hpp>
#include <utility>

#include "image/image_buffer.hpp"

namespace puerhlab {
ExposureOp::ExposureOp() : _exposure_offset(0.0f) {}

ExposureOp::ExposureOp(float exposure_offset) : _exposure_offset(exposure_offset) {}

auto ExposureOp::Apply(ImageBuffer& input) -> ImageBuffer {
  float          scale        = std::pow(2.0f, _exposure_offset);

  const cv::Mat& linear_image = input.GetCPUData();
  linear_image *= scale;

  return {std::move(linear_image)};
}

auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
}

};  // namespace puerhlab