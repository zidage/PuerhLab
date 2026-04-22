//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/edit_viewer/viewer_state.hpp"

class QWidget;

namespace alcedo {

struct ViewerViewState {
  ViewerStateSnapshot snapshot{};
  bool                prefer_interactive_primary = false;
  bool                allow_detail_patch        = true;
  bool                has_expected_detail_token = false;
  std::uint64_t       expected_detail_generation = 0;
  std::uint64_t       expected_detail_serial     = 0;
};

struct EditViewerRenderTargetResizeDecision {
  bool need_resize = false;
  int  slot_index  = 0;
};

struct EditViewerRenderTargetState {
  int                   slot_index         = 0;
  int                   width              = 0;
  int                   height             = 0;
  FramePresentationMode presentation_mode  = FramePresentationMode::FullFrame;
  FramePreviewMetadata  preview_metadata   = {};
};

class IEditViewerSurface {
 public:
  virtual ~IEditViewerSurface() = default;

  virtual auto widget() -> QWidget*                       = 0;
  virtual void submitFrame(const ViewerFrame& frame)      = 0;
#ifdef HAVE_METAL
  virtual void submitMetalFrame(const ViewerMetalFrame& frame) { (void)frame; }
#endif
  virtual void setDisplayConfig(const ViewerDisplayConfig& config) = 0;
  virtual void setViewState(const ViewerViewState& state) = 0;
  virtual void requestRedraw()                            = 0;
};

class IEditViewerRenderTargetSurface {
 public:
  virtual ~IEditViewerRenderTargetSurface() = default;

  virtual auto supportsDirectCudaPresent() const -> bool { return false; }
  virtual auto prepareRenderTarget(int width, int height)
      -> EditViewerRenderTargetResizeDecision {
    (void)width;
    (void)height;
    return {};
  }
  virtual void commitRenderTargetResize(int slot_index, int width, int height) {
    (void)slot_index;
    (void)width;
    (void)height;
  }
  virtual auto mapResourceForWrite() -> FrameWriteMapping { return {}; }
  virtual void unmapResource() {}
  virtual void notifyFrameReady() {}
  virtual void setNextFramePresentationMode(FramePresentationMode) {}
  virtual void setNextFramePreviewMetadata(const FramePreviewMetadata&) {}
  virtual auto activeRenderTargetState() const -> EditViewerRenderTargetState { return {}; }
  virtual auto hasRenderTarget(int slot_index, int width, int height) const -> bool = 0;
  virtual auto ensureRenderTarget(int slot_index, int width, int height) -> bool     = 0;
};

}  // namespace alcedo
