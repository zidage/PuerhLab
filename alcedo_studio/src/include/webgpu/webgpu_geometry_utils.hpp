//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_WEBGPU

#include "image/webgpu_image.hpp"

namespace alcedo {
namespace webgpu {
namespace utils {

void Rotate180(WebGpuImage& image);
void Rotate90CW(WebGpuImage& image);
void Rotate90CCW(WebGpuImage& image);

}  // namespace utils
}  // namespace webgpu
}  // namespace alcedo

#endif
