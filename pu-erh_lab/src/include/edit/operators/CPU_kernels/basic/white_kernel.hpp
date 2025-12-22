#pragma once
#include "edit/operators/op_base.hpp"

namespace puerhlab {
struct WhiteOpKernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    p.r = (p.r - params.black_point) * params.white_slope + params.white_point;
    p.g = (p.g - params.black_point) * params.white_slope + params.white_point;
    p.b = (p.b - params.black_point) * params.white_slope + params.white_point;
  }
};
}  // namespace puerhlab