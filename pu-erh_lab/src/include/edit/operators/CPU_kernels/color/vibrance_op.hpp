#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct VibranceOpKernel : PointOpTag {
  inline void operator()(Pixel& in, OperatorParams& params) const {
    float r = in.r, g = in.g, b = in.b;

    float max_val = std::max({r, g, b});
    float min_val = std::min({r, g, b});
    float chroma  = max_val - min_val;

    // chroma in [0, max], vibrance_offset in [-100, 100]
    float strength = params.vibrance_offset / 100.0f;

    // Protect already highly saturated color
    float falloff  = std::exp(-3.0f * chroma);

    float scale   = 1.0f + strength * falloff;

    if (params.vibrance_offset >= 0.0f) {
      float luma = r * 0.299f + g * 0.587f + b * 0.114f;

      r          = luma + (r - luma) * scale;
      g          = luma + (g - luma) * scale;
      b          = luma + (b - luma) * scale;

    } else {
      float avg = (r + g + b) / 3.0f;
      r += (avg - r) * (1.0f - scale);
      g += (avg - g) * (1.0f - scale);
      b += (avg - b) * (1.0f - scale);
    }

    in.r = r;
    in.g = g;
    in.b = b;
  }
};
}  // namespace puerhlab