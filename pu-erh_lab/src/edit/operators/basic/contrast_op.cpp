#include "edit/operators/basic/contrast_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <stdexcept>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
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

  linear_image.forEach<cv::Vec3f>([this](cv::Vec3f& pixel, const int*) -> void {
    auto lab = OklabCvt::ACESRGB2Oklab(pixel);
    lab.L    = (lab.L - 0.5f) * _scale + 0.5f;
    pixel    = OklabCvt::Oklab2ACESRGB(lab);
  });

  // linear_image          = (linear_image - 0.5f) * _scale + 0.5f;
  // clamp
  // cv::min(linear_image, 100.0f, linear_image);
  // cv::max(linear_image, 0.0f, linear_image);

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