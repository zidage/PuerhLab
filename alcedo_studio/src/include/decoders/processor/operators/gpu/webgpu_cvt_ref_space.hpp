//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_WEBGPU

#include "image/webgpu_image.hpp"

namespace alcedo {
namespace webgpu {

void Clamp01(WebGpuImage& image);
void ApplyInverseCamMulAndOrientRGBA(WebGpuImage& image, const float* cam_mul, int flip);

}  // namespace webgpu
}  // namespace alcedo

#endif
