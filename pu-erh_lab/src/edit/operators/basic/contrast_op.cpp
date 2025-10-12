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
void ContrastOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& linear_image = input->GetCPUData();

  linear_image.forEach<cv::Vec3f>([this](cv::Vec3f& pixel, const int*) -> void {
    auto lab = OklabCvt::ACESRGB2Oklab(pixel);
    lab.L    = (lab.L - 0.5f) * _scale + 0.5f;
    pixel    = OklabCvt::Oklab2ACESRGB(lab);
  });

  // linear_image          = (linear_image - 0.5f) * _scale + 0.5f;
  // clamp
  // cv::min(linear_image, 100.0f, linear_image);
  // cv::max(linear_image, 0.0f, linear_image);
}

auto ContrastOp::ToKernel() const -> Kernel {
  return Kernel {
    ._type = Kernel::Type::Point,
    ._func = PointKernelFunc([s=_scale](Pixel& in) {
      in.r = (in.r - 0.05707762557f) * s + 0.05707762557f; // 1 stop = 1/17.52
      in.g = (in.g - 0.05707762557f) * s + 0.05707762557f;
      in.b = (in.b - 0.05707762557f) * s + 0.05707762557f;
    })
  };
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