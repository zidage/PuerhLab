#include "edit/operators/basic/black_op.hpp"

#include <opencv2/core.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : _offset(offset) {
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), offset / 3.0f);
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point), _curve.white_point);
}

BlackOp::BlackOp(const nlohmann::json& params) {
  SetParams(params);
  _curve.black_point = hw::Set(hw::ScalableTag<float>(), _offset / 3.0f);
  _curve.white_point = hw::Set(hw::ScalableTag<float>(), 100.0f);
  _curve.slope       = hw::Div(hw::Sub(_curve.white_point, _curve.black_point), _curve.white_point);
}

void BlackOp::GetMask(cv::Mat& src, cv::Mat& mask) {}

auto BlackOp::GetOutput(cv::Vec3f& input) -> cv::Vec3f {
  float     y_intercept = _offset / 3.0f;
  float     white_point = 1.0f;

  float     slope       = (white_point - y_intercept) / 1.0f;

  cv::Vec3f output      = {
      slope * input[0] + y_intercept,
      slope * input[1] + y_intercept,
      slope * input[2] + y_intercept,
  };
  // output            = std::clamp(output, 0.0f, 100.0f);
  return output;
}

auto BlackOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  return hw::MulAdd(_curve.slope, luminance, _curve.black_point);
}

auto BlackOp::GetScale() -> float { return _offset / 3.0f; }

auto BlackOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<BlackOp>::Apply(input);
}

auto BlackOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void BlackOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>() / 100.0f;
  }
}
}  // namespace puerhlab