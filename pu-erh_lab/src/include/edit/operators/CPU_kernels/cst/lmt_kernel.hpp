#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct OCIO_LMT_Transform_Op : PointKernelFunc {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    // This kernel is a placeholder. The actual OCIO transform is applied in the Apply function.
    params.lmt_processor->applyRGBA(&p.r);
  }
};

}  // namespace puerhlab