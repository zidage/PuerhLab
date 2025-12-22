#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct TintOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.tint_enabled) return;
    p.g += params.tint_offset;
  }
};
}  // namespace puerhlab