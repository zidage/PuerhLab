#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct ShadowsOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    float L = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    if (L <= params.shadows_x0) {
      p.r = 0.0f;
      p.g = 0.0f;
      p.b = 0.0f;
    } else if (L < params.shadows_x1) {
      float t    = (L - params.shadows_x0) / params.shadows_dx;
      float H00  = 2 * t * t * t - 3 * t * t + 1;
      float H10  = t * t * t - 2 * t * t + t;
      float H01  = -2 * t * t * t + 3 * t * t;
      float H11  = t * t * t - t * t;
      float outL = H00 * params.shadows_y0 + H10 * (params.shadows_dx * params.shadows_m0) +
                   H01 * params.shadows_y1 + H11 * (params.shadows_dx * params.shadows_m1);

      if (!std::isfinite(outL)) outL = L;
      float scale = (L > 1e-8f) ? (outL / L) : 1.0f;
      p.r *= scale;
      p.g *= scale;
      p.b *= scale;
    }
  }
};
}  // namespace puerhlab