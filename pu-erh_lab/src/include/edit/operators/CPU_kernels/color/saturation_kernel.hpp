#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct SaturationOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.saturation_enabled) return;

    float luma = 0.2126f * p.r + 0.7152f * p.g + 0.0722f * p.b;
    p.r        = luma + (p.r - luma) * params.saturation_offset;
    p.g        = luma + (p.g - luma) * params.saturation_offset;
    p.b        = luma + (p.b - luma) * params.saturation_offset;
  }
};
}  // namespace puerhlab