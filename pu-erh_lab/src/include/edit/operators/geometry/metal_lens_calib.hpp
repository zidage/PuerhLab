//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_METAL

#include "edit/operators/geometry/lens_calib_runtime.hpp"

namespace puerhlab::metal {
class MetalImage;

void ApplyLensCalibration(MetalImage& image, const LensCalibGpuParams& params);

}  // namespace puerhlab::metal

#endif
