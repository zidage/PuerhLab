#include "edit/operators/basic/exposure_op.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <utility>

#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 */
ExposureOp::ExposureOp() : _exposure_offset(0.0f) { _scale = 0.0f; }

/**
 * @brief Construct a new Exposure Op:: Exposure Op object
 *
 * @param exposure_offset
 */
ExposureOp::ExposureOp(float exposure_offset) : _exposure_offset(exposure_offset) {
  // _scale        = std::pow(2.0f, _exposure_offset);
  _scale_factor = hw::Set(hw::ScalableTag<float>(), _scale);
}

ExposureOp::ExposureOp(const nlohmann::json& params) {
  SetParams(params);
  _scale_factor = hw::Set(hw::ScalableTag<float>(), _scale);
}

void ExposureOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    pixel[0] += _scale;
    pixel[1] += _scale;
    pixel[2] += _scale;
  });
}

auto ExposureOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[GetScriptName()] = _exposure_offset;

  return o;
}

void ExposureOp::SetParams(const nlohmann::json& params) {
  _exposure_offset = params[GetScriptName()];
  _scale           = _exposure_offset / 17.52f;
}

};  // namespace puerhlab