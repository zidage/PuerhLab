#pragma once
#include <algorithm>
namespace puerhlab {

inline float SmoothStep(float edge0, float edge1, float x) {
  float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
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
}  // namespace puerhlab
