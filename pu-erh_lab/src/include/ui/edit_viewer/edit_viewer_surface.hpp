//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/edit_viewer/viewer_state.hpp"

class QWidget;

namespace puerhlab {

struct ViewerViewState {
  ViewerStateSnapshot snapshot{};
};

class IEditViewerSurface {
 public:
  virtual ~IEditViewerSurface() = default;

  virtual auto widget() -> QWidget*                       = 0;
  virtual void submitFrame(const ViewerFrame& frame)      = 0;
  virtual void setViewState(const ViewerViewState& state) = 0;
  virtual void requestRedraw()                            = 0;
};

class IEditViewerRenderTargetSurface {
 public:
  virtual ~IEditViewerRenderTargetSurface() = default;

  virtual auto hasRenderTarget(int slot_index, int width, int height) const -> bool = 0;
  virtual auto ensureRenderTarget(int slot_index, int width, int height) -> bool     = 0;
};

}  // namespace puerhlab
