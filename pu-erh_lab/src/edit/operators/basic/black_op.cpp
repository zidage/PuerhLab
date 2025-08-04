#include "edit/operators/basic/black_op.hpp"

#include "image/image_buffer.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : _offset(offset) {}

BlackOp::BlackOp(const nlohmann::json& params) { SetParams(params); }

auto BlackOp::GetOutput(float luminance, float adj) -> float {
  return std::pow(1.0f - luminance, 5.0f);
}

auto BlackOp::GetScale() -> float { return _offset / 100.0f; }

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