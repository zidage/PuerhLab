#pragma once

#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct ExposureOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    p.r += params.exposure_offset;
    p.g += params.exposure_offset;
    p.b += params.exposure_offset;
  }
};
}  // namespace puerhlab