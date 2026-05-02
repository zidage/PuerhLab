//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <vector>

#include "ui/alcedo_main/editor_dialog/modules/curve.hpp"

namespace alcedo::ui {

struct ToneAdjustmentState {
  float exposure_ = 1.5f;
  float contrast_ = 0.0f;
  float blacks_ = 0.0f;
  float whites_ = 0.0f;
  float shadows_ = 0.0f;
  float highlights_ = 0.0f;
  std::vector<QPointF> curve_points_ = curve::DefaultCurveControlPoints();
  float saturation_ = 30.0f;
  float sharpen_ = 0.0f;
  float clarity_ = 0.0f;
};

}  // namespace alcedo::ui
