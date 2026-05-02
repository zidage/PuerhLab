//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>
#include <string>

#include "ui/alcedo_main/editor_dialog/modules/hls.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"

namespace alcedo::ui {

struct LookAdjustmentState {
  float hls_target_hue_ = 0.0f;
  float hls_hue_adjust_ = 0.0f;
  float hls_lightness_adjust_ = 0.0f;
  float hls_saturation_adjust_ = 0.0f;
  float hls_hue_range_ = hls::kDefaultHueRange;
  CdlWheelState lift_wheel_ = DefaultLiftWheelState();
  CdlWheelState gamma_wheel_ = DefaultGammaGainWheelState();
  CdlWheelState gain_wheel_ = DefaultGammaGainWheelState();
  hls::HlsProfileArray hls_hue_adjust_table_ = {};
  hls::HlsProfileArray hls_lightness_adjust_table_ = {};
  hls::HlsProfileArray hls_saturation_adjust_table_ = {};
  hls::HlsProfileArray hls_hue_range_table_ = hls::MakeFilledArray(hls::kDefaultHueRange);
  std::string lut_path_;
};

}  // namespace alcedo::ui
