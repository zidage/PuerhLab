//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "renderer/pipeline_task.hpp"
#include "ui/alcedo_main/editor_dialog/state/color_temp_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/display_transform_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/geometry_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/look_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/raw_decode_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/tone_adjustment_state.hpp"

namespace alcedo::ui {

struct EditorAdjustmentSnapshot {
  ToneAdjustmentState             tone_;
  ColorTempAdjustmentState        color_temp_;
  LookAdjustmentState             look_;
  DisplayTransformAdjustmentState display_transform_;
  GeometryAdjustmentState         geometry_;
  RawDecodeAdjustmentState        raw_;
  RenderType                      type_ = RenderType::FAST_PREVIEW;
};

}  // namespace alcedo::ui
