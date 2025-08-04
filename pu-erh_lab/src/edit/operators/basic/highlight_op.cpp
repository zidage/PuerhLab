#include "edit/operators/basic/highlight_op.hpp"

#include <cmath>

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
HighlightsOp::HighlightsOp(float offset) : _offset(offset) {}

HighlightsOp::HighlightsOp(const nlohmann::json& params) { SetParams(params); }

auto HighlightsOp::GetOutput(float luminance, float adj) -> float {
  if (luminance <= 0.7f) {
    return luminance;
  } else if (luminance > 0.7f && luminance <= 1.0f) {
    float term = 10.0f * luminance - 7.0f;
    return luminance + adj * luminance * std::pow(term, 1.1f) / 9.0f;
  } else {
    return 1.0f;
  }
}

auto HighlightsOp::GetScale() -> float { return _offset / 100.0f; }

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