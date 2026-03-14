//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/color_manager.hpp"

namespace puerhlab {

auto ColorManager::ApplyWindowColorSpace(void* native_view_or_window,
                                         const ViewerDisplayConfig& config) -> bool {
  (void)native_view_or_window;
  (void)config;
  return false;
}

}  // namespace puerhlab
