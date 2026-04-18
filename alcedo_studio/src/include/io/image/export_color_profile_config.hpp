//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "edit/operators/utils/color_utils.hpp"

namespace alcedo {

struct ExportColorProfileConfig {
  ColorUtils::ColorSpace encoding_space = ColorUtils::ColorSpace::REC709;
  ColorUtils::EOTF       encoding_eotf  = ColorUtils::EOTF::GAMMA_2_2;
  float                  peak_luminance = 100.0f;
};

}  // namespace alcedo
