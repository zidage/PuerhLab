#include "edit/operators/basic/black_op.hpp"

#include <opencv2/core.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "image/image_buffer.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : _offset(offset) {
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), offset / 3.0f);
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point), _curve.white_point);
}

BlackOp::BlackOp(const nlohmann::json& params) {
  SetParams(params);
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), _offset / 10.0f);
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point), _curve.white_point);
}

void BlackOp::GetMask(cv::Mat& src, cv::Mat& mask) {}

auto BlackOp::GetOutput(cv::Vec3f& input) -> cv::Vec3f {
  cv::Vec3f output = {
      _slope * input[0] + _y_intercept,
      _slope * input[1] + _y_intercept,
      _slope * input[2] + _y_intercept,
  };
  // output            = std::clamp(output, 0.0f, 100.0f);
  return output;
}

auto BlackOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  return hw::MulAdd(_curve.slope, luminance, _curve.black_point);
}

auto BlackOp::GetScale() -> float { return _offset / 3.0f; }

void BlackOp::Apply(std::shared_ptr<ImageBuffer> input) { ToneRegionOp<BlackOp>::Apply(input); }

auto BlackOp::ToKernel() const -> Kernel {
  return Kernel{._type = Kernel::Type::Point,
                ._func = PointKernelFunc([&a = _slope, &b = _y_intercept](Pixel& in) {
                  // return Pixel{in.r * s + b, in.g * s + b, in.b * s + b};
                  in.r = in.r * a + b;
                  in.g = in.g * a + b;
                  in.b = in.b * a + b;
                })};
}

auto BlackOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void BlackOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset      = 0.0f;
    _y_intercept = 0.0f;
    _slope       = 1.0f;

  } else {
    _offset      = params[_script_name].get<float>() / 100.0f;
    _y_intercept = _offset / 10.f;
    _slope       = (1.0f - _y_intercept) / 1.0f;
  }
  _y_intercept_vec = simple_simd::set1_f32(_y_intercept);
  _slope_vec       = simple_simd::set1_f32(_slope);
}
}  // namespace puerhlab