#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct TintOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const { p.g += params.tint_offset; }
};
}  // namespace puerhlab