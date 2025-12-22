#pragma once
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct WhiteOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.white_enabled) return;
    p.r = (p.r) * params.slope + params.black_point;
    p.g = (p.g) * params.slope + params.black_point;
    p.b = (p.b) * params.slope + params.black_point;
  }
};
}  // namespace puerhlab