#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/utils/functions.hpp"
#include "hwy/contrib/math/math-inl.h"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {
  _gamma  = std::pow(2.2f, -_offset / 100.0f);
  v_gamma = hw::Set(d, _gamma);
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  _gamma  = std::pow(2.2f, -_offset / 100.0f);
  v_gamma = hw::Set(d, _gamma);
}

auto ShadowsOp::GetOutput(hw::Vec<hw::ScalableTag<float>> luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  auto v_L_in       = hw::Div(luminance, _max);
  v_L_in            = hw::Max((v_L_in + _pivot) / (v_one - _pivot), v_zero);
  auto v_alpha      = hw::Max(v_zero, hw::Sub(v_one, hw::Mul(v_L_in, v_inv_threshold)));
  auto v_L_in_safe  = hw::Max(v_L_in, v_epsilon);
  auto v_log_L      = hw::Log(d, v_L_in_safe);
  auto v_exponent   = hw::Mul(v_gamma, v_log_L);
  auto v_L_adjusted = hw::Exp(d, v_exponent);
  auto v_diff       = hw::Sub(v_L_adjusted, v_L_in);
  auto v_L_out      = hw::MulAdd(v_alpha, v_diff, v_L_in);

  return hw::Mul(v_L_out, _max);
}

auto ShadowsOp::GetOutput(float luminance) -> float {
  float L_in       = luminance / 100.0f;

  L_in             = std::max((L_in + 0.05f) / (1.0f - 0.05f), 0.0f);

  float alpha      = std::max(0.0f, 1.0f - L_in * _inv_threshold);

  float L_in_safe  = std::max(L_in, 1e-7f);
  float log_L      = std::log(L_in_safe);
  float exponent   = _gamma * log_L;
  float L_adjusted = std::exp(exponent);
  float diff       = L_adjusted - L_in;
  float L_out      = alpha * diff + L_in;
  return L_out * 100.0f;
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