#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include "edit/operators/op_base.hpp"

namespace puerhlab {

struct OCIO_LMT_Transform_Op_Kernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    if (!params.lmt_enabled || !params.cpu_lmt_processor) return;
    params.cpu_lmt_processor->applyRGBA(&p.r);
  }
};

}  // namespace puerhlab