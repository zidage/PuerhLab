#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct ContrastOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.contrast_enabled) return;
    p.r = (p.r - 0.05707762557f) * params.contrast_scale + 0.05707762557f;  // 1 stop = 1/17.52
    p.g = (p.g - 0.05707762557f) * params.contrast_scale + 0.05707762557f;
    p.b = (p.b - 0.05707762557f) * params.contrast_scale + 0.05707762557f;
  }
};
}  // namespace puerhlab