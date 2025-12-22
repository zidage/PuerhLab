#include "edit/operators/color/saturation_op.hpp"

#include <memory>
#include <opencv2/core/mat.hpp>
#include <utility>

#include "edit/operators/color/conversion/Oklab_cvt.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/operator_factory.hpp"
#include "json.hpp"

namespace puerhlab {

SaturationOp::SaturationOp() : _saturation_offset(0) { ComputeScale(); }

SaturationOp::SaturationOp(float saturation_offset) : _saturation_offset(saturation_offset) {
  ComputeScale();
}

SaturationOp::SaturationOp(const nlohmann::json& params) { SetParams(params); }

/**
 * @brief Compute the scale from the offset
 *
 */
void SaturationOp::ComputeScale() {
  if (_saturation_offset >= 0.0f) {
    _scale = 1.0f + _saturation_offset / 100.0f;
  } else {
    _scale = 1.0f + _saturation_offset / 100.0f;
  }
}

void SaturationOp::Apply(std::shared_ptr<ImageBuffer> input) {
  cv::Mat& img = input->GetCPUData();

  img.forEach<cv::Vec3f>([&](cv::Vec3f& pixel, const int*) {
    OklabCvt::Oklab oklab_vec = OklabCvt::LinearRGB2Oklab(pixel);

    // Chroma = a^2 + b^2
    oklab_vec.a *= _scale;
    oklab_vec.b *= _scale;

    pixel = OklabCvt::Oklab2LinearRGB(oklab_vec);
  });
}

auto SaturationOp::ToKernel() const -> Kernel {
  return Kernel{._type = Kernel::Type::Point, ._func = PointKernelFunc([&s = _scale](Pixel& in) {
                                                // OklabCvt::Oklab oklab_vec =
                                                // OklabCvt::ACESRGB2Oklab(in);

                                                float luma = 0.2126f * in.r + 0.7152f * in.g +
                                                             0.0722f * in.b;
                                                in.r = luma + (in.r - luma) * s;
                                                in.g = luma + (in.g - luma) * s;
                                                in.b = luma + (in.b - luma) * s;
                                                // Chroma = a^2 + b^2
                                                // oklab_vec.a *= s;
                                                // oklab_vec.b *= s;

                                                // OklabCvt::Oklab2ACESRGB(oklab_vec, in);
                                              })};
}

auto SaturationOp::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  o[_script_name] = _saturation_offset;

  return o;
}

void SaturationOp::SetParams(const nlohmann::json& params) {
  if (params.contains(_script_name)) {
    _saturation_offset = params[_script_name];
  } else {
    _saturation_offset = 0.0f;
  }
  ComputeScale();
}

void SaturationOp::SetGlobalParams(OperatorParams& params) const {
  params.saturation_offset = _scale;
}
};  // namespace puerhlab