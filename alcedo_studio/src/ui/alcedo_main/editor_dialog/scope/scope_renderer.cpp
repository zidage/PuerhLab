//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/scope/scope_renderer.hpp"

#include <QtGlobal>

#include <algorithm>
#include <cmath>

namespace alcedo::ui {
namespace {

auto DensityToByte(float value) -> int {
  const float shaped = std::pow(std::clamp(value, 0.0f, 1.0f), 0.55f);
  return static_cast<int>(std::round(shaped * 255.0f));
}

}  // namespace

auto ScopeRenderer::Render(const ScopeOutputSet& output) const -> ScopePresentation {
  ScopePresentation presentation;
  const ScopeRenderSnapshot snapshot = ReadScopeRenderSnapshot(output);
  presentation.generation            = snapshot.generation;

  if (snapshot.histogram.valid) {
    presentation.histogram.bins                   = snapshot.histogram.bins;
    presentation.histogram.clip_tail_bins         = snapshot.histogram.clip_tail_bins;
    presentation.histogram.shadow_clip_ratio      = snapshot.histogram.shadow_clip_ratio;
    presentation.histogram.highlight_clip_ratio   = snapshot.histogram.highlight_clip_ratio;
    presentation.histogram.shadow_clip_warning    = snapshot.histogram.shadow_clip_warning;
    presentation.histogram.highlight_clip_warning = snapshot.histogram.highlight_clip_warning;
    presentation.histogram.rgb                    = snapshot.histogram.rgb;
    presentation.histogram.valid                  = true;
  }

  if (snapshot.waveform.valid) {
    QImage image(snapshot.waveform.width, snapshot.waveform.height, QImage::Format_ARGB32_Premultiplied);
    image.fill(qRgba(0, 0, 0, 255));

    for (int y = 0; y < snapshot.waveform.height; ++y) {
      for (int x = 0; x < snapshot.waveform.width; ++x) {
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(snapshot.waveform.width) +
                            static_cast<size_t>(x)) *
                           4U;
        const int r = DensityToByte(snapshot.waveform.rgba[idx + 0]);
        const int g = DensityToByte(snapshot.waveform.rgba[idx + 1]);
        const int b = DensityToByte(snapshot.waveform.rgba[idx + 2]);
        const int a = std::max({r, g, b, 18});
        image.setPixel(x, y, qRgba(r, g, b, a));
      }
    }

    presentation.waveform.image = image;
    presentation.waveform.valid = true;
  }

  return presentation;
}

}  // namespace alcedo::ui
