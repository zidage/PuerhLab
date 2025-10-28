#include "edit/operators/basic/shadow_op.hpp"

#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "edit/operators/basic/tone_region_op.hpp"
#include "edit/operators/op_kernel.hpp"
#include "edit/operators/utils/functions.hpp"
#include "hwy/contrib/math/math-inl.h"
#include "image/image_buffer.hpp"

namespace puerhlab {
ShadowsOp::ShadowsOp(float offset) : _offset(offset) {
  float normalized_offset = _offset / 100.0f;
  _gamma = std::pow(2.0f, -normalized_offset * 1.3f);
  v_gamma = hw::Set(d, _gamma);
  // InitializeLUT();
}

ShadowsOp::ShadowsOp(const nlohmann::json& params) {
  SetParams(params);
  float normalized_offset = _offset / 100.0f;
  _gamma = std::pow(2.0f, -normalized_offset * 1.3f);
  v_gamma = hw::Set(d, _gamma);
  // InitializeLUT();
}

void ShadowsOp::InitializeLUT() {
  _lut.resize(kLutSize);
  for (int i = 0; i < kLutSize; ++i) {
    float x = static_cast<float>(i) / (kLutSize - 1);
    float y = (x == 0.0f && _gamma < 0.0f) ? 0.0f : std::pow(x, _gamma);
    _lut[i] = y;
  }
}

auto ShadowsOp::GetOutput(hw::Vec<hw::ScalableTag<float>>& luminance)
    -> hw::Vec<hw::ScalableTag<float>> {
  auto v_L_in         = hw::Mul(luminance, v_inv_max);
  v_L_in              = hw::MulAdd(v_L_in, v_pivot_scale, v_pivot_offset);
  v_L_in              = hw::Max(v_L_in, v_zero);

  // alpha = max(1 - L_in / threshold, 0)
  auto v_alpha        = hw::NegMulAdd(v_L_in, v_inv_threshold, v_one);
  v_alpha             = hw::Max(v_alpha, v_zero);

  const auto v_f_idx  = hw::Mul(v_L_in, v_lut_scale);
  const auto v_idx0_f = hw::Floor(v_f_idx);          // float
  const auto v_weight = hw::Sub(v_f_idx, v_idx0_f);  // in [0,1)

  const auto v_i_idx0 = hw::ConvertTo(di, v_idx0_f);  // int32
  const auto v_i_idx1 = hw::Min(hw::Add(v_i_idx0, hw::Set(di, 1)), hw::Set(di, kLutSize - 1));

  const float* HWY_RESTRICT lut          = _lut.data();
  const auto                v_y0         = hw::GatherIndex(d, lut, v_i_idx0);
  const auto                v_y1         = hw::GatherIndex(d, lut, v_i_idx1);

  const auto                v_L_adjusted = hw::MulAdd(v_weight, hw::Sub(v_y1, v_y0), v_y0);
  const auto                v_diff       = hw::Sub(v_L_adjusted, v_L_in);
  const auto                v_L_out      = hw::MulAdd(v_alpha, v_diff, v_L_in);

  return hw::Mul(v_L_out, _max);
}

auto ShadowsOp::GetOutput(cv::Vec3f& input) -> cv::Vec3f {
  // float l_in       = luminance / 100.0f;
  cv::Vec3f output = {};
  for (int c = 0; c < 3; ++c) {
    float l_in       = (input[c] + 0.05f) / (1.0f - 0.05f);
    l_in             = std::fmax(l_in, 0.0f);

    float alpha      = 1.0f - l_in * _inv_threshold;
    alpha            = std::fmax(alpha, 0.0f);

    float l_in_safe  = std::fmax(l_in, 1e-7f);  // Avoid division by zero
    float l_adjusted = std::pow(l_in_safe, _gamma);

    float diff       = l_adjusted - l_in;
    output[c]        = l_in + alpha * diff;
  }

  return output;
}

auto ShadowsOp::GetScale() -> float { return _offset / 100.0f; }

void ShadowsOp::Apply(std::shared_ptr<ImageBuffer> input) { ToneRegionOp<ShadowsOp>::Apply(input); }

auto ShadowsOp::ToKernel() const -> Kernel {
  return Kernel{._type = Kernel::Type::Point,
                ._func = PointKernelFunc([&inv = _inv_threshold, &g = _gamma](Pixel& in) {
                  constexpr float kPivot = 0.05f;
                  constexpr float eps = 1e-7f;
                  constexpr float threshold = 0.2f;

                  const float L = 0.2126f * in.r + 0.7152f * in.g + 0.0722f * in.b;
                  
                  if (L < eps) {
                    return;
                  }

                  float Lp = (L + 0.05f) / (1.0f - kPivot);
                  Lp = std::fmax(Lp, 0.0f);

                  float alpha = 1.0f - Lp / threshold;
                  alpha = std::fmax(alpha, 0.0f);

                  const float Lp_adjust = std::pow(Lp, g);
                  const float Lp_out = Lp + alpha * (Lp_adjust - Lp);
                  const float L_out = (Lp_out * (1.0f - kPivot) - kPivot);

                  const float scale = L_out / L;

                  in.r *= scale;
                  in.g *= scale;
                  in.b *= scale;
                })};
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