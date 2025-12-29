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
    // The pair of transform ops should always be enabled.
    if (params.is_working_space) {
      params.cpu_to_working_processor->applyRGBA(&p.r);
      params.is_working_space = false;
    } else {
      params.cpu_to_output_processor->applyRGBA(&p.r);
      params.is_working_space = true;
    }
  }
};

}  // namespace puerhlab