//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>

#include "edit/operators/op_base.hpp"
#include "edit/scope/scope_analyzer.hpp"
#include "ui/edit_viewer/frame_sink.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"

namespace alcedo {
class QtEditViewer;
class CPUPipelineExecutor;
class FinalDisplayFrameTapSink;
}  // namespace alcedo

namespace alcedo::ui {

class ScopePanel;

class EditorFrameManager final {
 public:
  EditorFrameManager() = default;
  ~EditorFrameManager();

  void SetViewer(QtEditViewer* viewer);
  void SetScopePanel(ScopePanel* scope_panel);

  void EnsureFrameRouting();
  void AttachExecutionStages(const std::shared_ptr<CPUPipelineExecutor>& exec);
  auto CurrentFrameSink() -> IFrameSink*;

  void SyncViewerDisplayEncoding(ColorUtils::ColorSpace encoding_space,
                                 ColorUtils::EOTF       encoding_eotf);

  void MarkNeedsFullFramePreviewAfterGeometryCommit();
  auto NeedsFullFramePreviewAfterGeometryCommit() const -> bool;
  auto UseViewportRegionForPanelChange(ControlPanelKind previous_panel,
                                       ControlPanelKind next_panel) -> bool;

 private:
  QtEditViewer*                         viewer_      = nullptr;
  ScopePanel*                           scope_panel_ = nullptr;
  std::shared_ptr<IScopeAnalyzer>       scope_analyzer_{};
  std::unique_ptr<FinalDisplayFrameTapSink> final_display_frame_tap_{};
  bool                                  force_next_full_frame_preview_ = false;
};

}  // namespace alcedo::ui
