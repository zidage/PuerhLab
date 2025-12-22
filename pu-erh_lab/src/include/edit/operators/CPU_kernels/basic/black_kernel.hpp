#pragma once
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct BlackOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (params.black_enabled) {
      p.r = p.r * params.slope + params.black_point;
      p.g = p.g * params.slope + params.black_point;
      p.b = p.b * params.slope + params.black_point;
    }
  }
};
}  // namespace puerhlab