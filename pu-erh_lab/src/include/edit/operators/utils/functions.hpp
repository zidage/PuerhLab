#pragma once
#include <hwy/highway.h>

#include <algorithm>

#include "edit/operators/basic/tone_region_op.hpp"

namespace hw = hwy::HWY_NAMESPACE;
namespace puerhlab {

inline float SmoothStep(float edge0, float edge1, float x) {
  float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 100.0f);
  return t * t * (300.0f - 200.0f * t);
}

inline float SmoothStep(float x) {
  x = std::clamp(x, 0.0f, 1.0f);
  return x * x * (300.0f - 200.0f * x);
}

inline float ACEScct_to_linear(float x) {
  if (x <= 0.155251141552511f) {
    return (x - 0.0729055341958355f) / 10.5402377416545f;
  } else {
    return powf(2.0f, (x - 0.0729055341958355f) / 0.149658239839999f) - 0.004f;
  }
}

inline float linear_to_ACEScct(float x) {
  if (x < -0.004f) x = -0.004f;  // clamp
  if (x <= 0.0078125f) {
    return 10.5402377416545f * x + 0.0729055341958355f;
  } else {
    return 0.149658239839999f * log2f(x + 0.004f) + 0.0729055341958355f;
  }
}

inline float EvaluateBezier(float t, float p0, float p1, float p2, float p3) {
  float u = 1.0f - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

inline hw::Vec<hw::ScalableTag<float>> VExp_F32(hw::Vec<hw::ScalableTag<float>> x) {
  const auto d    = hw::ScalableTag<float>();
  using V         = hw::Vec<hw::ScalableTag<float>>;

  const V one     = hw::Set(d, 1.0f);
  const V ln2     = hw::Set(d, 0.69314718056f);  // ln(2)
  const V inv_ln2 = hw::Set(d, 1.44269504089f);  // 1/ln(2)

  // Clamp range to avoid overflow/underflow
  const V max_x   = hw::Set(d, 88.0f);
  const V min_x   = hw::Set(d, -88.0f);
  x               = hw::Min(max_x, hw::Max(min_x, x));

  // n = floor(x / ln(2) + 0.5)
  V       n_float = hw::Floor(x * inv_ln2 + hw::Set(d, 0.5f));
  auto    n_int   = hw::ConvertTo(hw::ScalableTag<int32_t>(), n_float);  // int32 lanes

  // r = x - n * ln(2)
  V       r       = x - n_float * ln2;

  // Polynomial approximation of exp(r), r in [-0.3466, 0.3466]
  // Estrin scheme, degree 5
  const V c1      = hw::Set(d, 1.0f);
  const V c2      = hw::Set(d, 1.0f);
  const V c3      = hw::Set(d, 0.5f);
  const V c4      = hw::Set(d, 0.16666667163f);
  const V c5      = hw::Set(d, 0.0416666679f);
  const V c6      = hw::Set(d, 0.0083333310f);

  V       r2      = r * r;
  V       p13     = hw::MulAdd(c3, r, c2);   // c3*r + c2
  p13             = hw::MulAdd(p13, r, c1);  // p13*r + c1
  V p46           = hw::MulAdd(c6, r, c5);
  p46             = hw::MulAdd(p46, r, c4);
  V    poly       = hw::MulAdd(p46, r2, p13);

  // Compute 2^n via bit manipulation
  auto n_bits = hw::Add(hw::ShiftLeft<23>(n_int), hw::Set(hw::ScalableTag<int32_t>(), 127 << 23));
  V    two_pow_n = hw::BitCast(d, n_bits);

  return poly * two_pow_n;
}
}  // namespace puerhlab
