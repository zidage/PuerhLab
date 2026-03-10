//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <libraw/libraw.h>

#include "image/metal_image.hpp"

namespace puerhlab {
namespace metal {
void BayerRGGB2RGB_RCD(MetalImage& image);
};
};