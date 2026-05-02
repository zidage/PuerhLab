//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/alcedo_main/editor_dialog/modules/geometry.hpp"

namespace alcedo::ui {

struct GeometryAdjustmentState {
  float rotate_degrees_ = 0.0f;
  bool crop_enabled_ = true;
  float crop_x_ = 0.0f;
  float crop_y_ = 0.0f;
  float crop_w_ = 1.0f;
  float crop_h_ = 1.0f;
  bool crop_expand_to_fit_ = true;
  geometry::CropAspectPreset crop_aspect_preset_ = geometry::CropAspectPreset::Free;
  float crop_aspect_width_ = 1.0f;
  float crop_aspect_height_ = 1.0f;
};

}  // namespace alcedo::ui
