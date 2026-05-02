//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/alcedo_main/editor_dialog/state.hpp"

namespace alcedo::ui {

// Phase 1 keeps the legacy AdjustmentState as the render and commit snapshot.
// Later phases split this compatibility wrapper into per-module typed states.
struct EditorAdjustmentSnapshot {
  AdjustmentState state{};
  RenderType      type = RenderType::FAST_PREVIEW;
};

}  // namespace alcedo::ui
