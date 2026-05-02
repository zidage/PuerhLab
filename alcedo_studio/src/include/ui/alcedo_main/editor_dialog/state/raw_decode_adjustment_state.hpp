//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <string>

namespace alcedo::ui {

struct RawDecodeAdjustmentState {
  bool raw_highlights_reconstruct_ = true;
  bool lens_calib_enabled_ = true;
  std::string lens_override_make_{};
  std::string lens_override_model_{};
};

}  // namespace alcedo::ui
