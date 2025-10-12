#include "edit/operators/basic/white_op.hpp"

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
WhiteOp::WhiteOp(float offset) : _offset(offset) {
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f + offset / 3.0f);
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), 0.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point),
                               hw::Set(hw::ScalableTag<float>(), 100.0f));
}

WhiteOp::WhiteOp(const nlohmann::json& params) {
  SetParams(params);
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f + _offset / 3.0f);
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), 0.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point),
                               hw::Set(hw::ScalableTag<float>(), 100.0f));
}

void WhiteOp::GetMask(cv::Mat& src, cv::Mat& mask) {}

auto WhiteOp::GetOutput(cv::Vec3f& input) -> cv::Vec3f {
  cv::Vec3f output      = {
      _slope * input[0],
      _slope * input[1],
      _slope * input[2],
  };
  // output            = std::clamp(output, 0.0f, 100.0f);
  return output;
}

auto WhiteOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  return hw::MulAdd(_curve.slope, luminance, _curve.black_point);
}

auto WhiteOp::GetScale() -> float { return _offset / 3.0f; }

void WhiteOp::Apply(std::shared_ptr<ImageBuffer> input) {
  ToneRegionOp<WhiteOp>::Apply(input);
}

auto WhiteOp::ToKernel() const -> Kernel {
  return Kernel {
    ._type = Kernel::Type::Point,
    ._func = PointKernelFunc([s=_slope](Pixel& in) {
      in.r = in.r * s;
      in.g = in.g * s;
      in.b = in.b * s;
    })
  };
}

auto WhiteOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void WhiteOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
    _y_intercept = 1.0f;
    _black_point = 0.0f;
    _slope = 1.0f;
  } else {
    _offset = params[_script_name].get<float>();
    _y_intercept = 1.0f + _offset / 300.0f;
    _black_point = 0.0f;  
    _slope = (_y_intercept - _black_point) / 1.0f;
  }
}
}  // namespace puerhlab