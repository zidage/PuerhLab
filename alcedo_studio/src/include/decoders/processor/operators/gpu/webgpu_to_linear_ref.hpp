//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_WEBGPU

#include <libraw/libraw.h>

#include "decoders/processor/raw_processor_pattern.hpp"
#include "image/webgpu_image.hpp"

namespace alcedo {
namespace webgpu {

void ToLinearRef(webgpu::WebGpuImage& img, LibRaw& raw_processor, const RawCfaPattern& pattern);

}  // namespace webgpu
}  // namespace alcedo

#endif
