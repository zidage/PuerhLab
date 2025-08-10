#include "edit/operators/basic/black_op.hpp"

#include "image/image_buffer.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : _offset(offset) {}

BlackOp::BlackOp(const nlohmann::json& params) { SetParams(params); }

void BlackOp::GetMask(cv::Mat& src, cv::Mat& mask) {}

auto BlackOp::GetOutput(float luminance, float adj) -> float {
  float y_intercept = adj;
  float white_point = 100.0f;

  float slope       = (white_point - y_intercept) / 100.0f;

  float output      = slope * luminance + y_intercept;
  output            = std::clamp(output, 0.0f, 100.0f);
  return output;
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
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab