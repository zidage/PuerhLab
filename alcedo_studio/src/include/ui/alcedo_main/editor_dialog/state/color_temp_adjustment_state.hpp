//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/alcedo_main/editor_dialog/state.hpp"

namespace alcedo::ui {

struct ColorTempAdjustmentState {
  ColorTempMode mode_ = ColorTempMode::AS_SHOT;
  float custom_cct_ = 6500.0f;
  float custom_tint_ = 0.0f;
  float resolved_cct_ = 6500.0f;
  float resolved_tint_ = 0.0f;
  bool supported_ = true;
};

}  // namespace alcedo::ui
