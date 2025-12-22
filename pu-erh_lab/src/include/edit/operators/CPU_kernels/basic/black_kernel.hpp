#pragma once
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct BlackOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    p.r = p.r + params.black_point;
    p.g = p.g + params.black_point;
    p.b = p.b + params.black_point;
  }
};
}  // namespace puerhlab