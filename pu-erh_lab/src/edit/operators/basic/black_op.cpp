#include "edit/operators/basic/black_op.hpp"

#include <opencv2/core.hpp>

#include "image/image_buffer.hpp"
#include "utils/simd/simple_simd.hpp"

namespace puerhlab {
BlackOp::BlackOp(float offset) : _offset(offset) {}

BlackOp::BlackOp(const nlohmann::json& params) { SetParams(params); }

auto BlackOp::GetScale() -> float { return _offset / 3.0f; }

void BlackOp::Apply(std::shared_ptr<ImageBuffer> input) { (void)input; }


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

void BlackOp::SetGlobalParams(OperatorParams& params) const {
  // Should only be called once SetParams has been called
  params.black_point = _y_intercept;
  params.slope       = (params.white_point - params.black_point);
}
}  // namespace puerhlab