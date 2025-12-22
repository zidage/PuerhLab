#pragma once

#include <OpenColorIO/OpenColorIO.h>
#include <OpenColorIO/OpenColorTypes.h>

#include <filesystem>
#include <optional>

#include "edit/operators/op_base.hpp"
#include "edit/operators/op_kernel.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab {
struct OCIO_ACES_Transform_Op_Kernel : PointOpTag {
  inline void operator()(Pixel& p, OperatorParams& params) const {
    // This kernel is a placeholder. The actual OCIO transform is applied in the Apply function.
    params.lmt_processor->applyRGBA(&p.r);
  }
 };

}  // namespace puerhlab