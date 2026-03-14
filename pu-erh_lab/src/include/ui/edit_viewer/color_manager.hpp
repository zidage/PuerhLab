//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {

class ColorManager {
 public:
  static auto ApplyWindowColorSpace(void* native_view_or_window,
                                    const ViewerDisplayConfig& config) -> bool;
};

}  // namespace puerhlab
