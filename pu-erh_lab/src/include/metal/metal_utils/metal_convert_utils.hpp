//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_METAL

namespace puerhlab::metal {
class MetalImage;
}

namespace puerhlab::metal::utils {

void ConvertTexture(const MetalImage& src, MetalImage& dst, double alpha = 1.0, double beta = 0.0);

}  // namespace puerhlab::metal::utils

#endif
