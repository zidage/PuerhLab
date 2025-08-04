#include "edit/operators/basic/shadow_op.hpp"

#include "edit/operators/utils/functions.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {}

ShadowsOp::ShadowsOp(const nlohmann::json& params) { SetParams(params); }

auto ShadowsOp::GetOutput(float luminance, float adj) -> float {
  float x = luminance;
  return x + adj * x * std::pow(1 - x, 2.0f) * std::exp(-40 * std::pow(x - 0.2f, 2.0f));
}

auto ShadowsOp::GetScale() -> float { return _offset / 100.0f; }

auto ShadowsOp::Apply(ImageBuffer& input) -> ImageBuffer {
  return ToneRegionOp<ShadowsOp>::Apply(input);
}

auto ShadowsOp::GetParams() const -> nlohmann::json { return {_script_name, _offset}; }

void ShadowsOp::SetParams(const nlohmann::json& params) {
  if (!params.contains(_script_name)) {
    _offset = 0.0f;
  } else {
    _offset = params[_script_name].get<float>();
  }
}
}  // namespace puerhlab