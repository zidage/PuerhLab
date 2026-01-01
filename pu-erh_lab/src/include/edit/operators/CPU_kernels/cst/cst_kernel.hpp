//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

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
    if (params.is_working_space_) {
      params.cpu_to_working_processor_->applyRGBA(&p.r_);
      params.is_working_space_ = false;
    } else {
      params.cpu_to_output_processor_->applyRGBA(&p.r_);
      params.is_working_space_ = true;
    }
  }
};

}  // namespace puerhlab