#include "edit/operators/basic/highlight_op.hpp"

#include <easy/profiler.h>

#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
HighlightsOp::HighlightsOp(float offset) : _offset(offset) {
  _ctrl_param = hw::Mul(hw::Set(hw::ScalableTag<float>(), offset / 100.0f),
                        hw::Set(hw::ScalableTag<float>(), 40.0f));
}

HighlightsOp::HighlightsOp(const nlohmann::json& params) {
  SetParams(params);
  _ctrl_param = hw::Mul(hw::Set(hw::ScalableTag<float>(), _offset / 100.0f),
                        hw::Set(hw::ScalableTag<float>(), 40.0f));
}

auto HighlightsOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  auto scaled_luminance = hw::Div(luminance, hw::Set(hw::ScalableTag<float>(), 100.0f));
  auto t_square         = hw::Mul(scaled_luminance, scaled_luminance);

  auto result           = hw::MulAdd(_ctrl_param, t_square, luminance);
  return result;
}

auto HighlightsOp::GetOutput(float luminance) -> float {
  float scaled_luminance = luminance / 100.0f;
  float t_square         = scaled_luminance * scaled_luminance;

  return (_offset / 100.0f * 40.0f) * t_square + luminance;
}

auto HighlightsOp::GetScale() -> float { return _offset / 300.0f; }

auto HighlightsOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<HighlightsOp>::Apply(input);
}

auto HighlightsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void HighlightsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab