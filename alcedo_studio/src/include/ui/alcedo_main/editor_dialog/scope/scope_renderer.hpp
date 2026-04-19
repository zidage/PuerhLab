//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QImage>

#include <cstdint>
#include <vector>

#include "edit/scope/scope_analyzer.hpp"

namespace alcedo::ui {

struct ScopeHistogramPresentation {
  int                bins                   = 0;
  int                clip_tail_bins         = 0;
  float              shadow_clip_ratio      = 0.0f;
  float              highlight_clip_ratio   = 0.0f;
  bool               shadow_clip_warning    = false;
  bool               highlight_clip_warning = false;
  std::vector<float> rgb                    = {};
  bool               valid                  = false;
};

struct ScopeWaveformPresentation {
  QImage image = {};
  bool   valid = false;
};

struct ScopePresentation {
  ScopeHistogramPresentation histogram = {};
  ScopeWaveformPresentation  waveform  = {};
  uint64_t                   generation = 0;
};

class ScopeRenderer {
 public:
  auto Render(const ScopeOutputSet& output) const -> ScopePresentation;
};

}  // namespace alcedo::ui
