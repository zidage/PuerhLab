#include "edit/operators/basic/white_op.hpp"

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
WhiteOp::WhiteOp(float offset) : _offset(offset) {}

WhiteOp::WhiteOp(const nlohmann::json& params) { SetParams(params); }

void WhiteOp::GetMask(cv::Mat& src, cv::Mat& mask) {}

auto WhiteOp::GetOutput(float luminance, float adj) -> float { return std::pow(luminance, 4.0f); }

auto WhiteOp::GetScale() -> float { return _offset / 100.0f; }

auto WhiteOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<WhiteOp>::Apply(input);
}

auto WhiteOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void WhiteOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab