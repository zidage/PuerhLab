//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <mutex>

#include "edit/scope/scope_analyzer.hpp"
#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {

class FinalDisplayFrameTapSink final : public IFrameSink, public IFinalDisplayFrameProvider {
 public:
  FinalDisplayFrameTapSink(IFrameSink* downstream_sink,
                           std::shared_ptr<IScopeAnalyzer> scope_analyzer);

  void SetScopeRequest(const ScopeRequest& request);
  auto GetScopeRequest() const -> ScopeRequest;

  auto GetCurrentDisplayFrameView() const -> FinalDisplayFrameView override;

  void EnsureSize(int width, int height) override;
  auto MapResourceForWrite() -> FrameWriteMapping override;
  void UnmapResource() override;
  void NotifyFrameReady() override;
  void SubmitHostFrame(const ViewerFrame& frame) override;
#ifdef HAVE_METAL
  void SubmitMetalFrame(const ViewerMetalFrame& frame) override;
#endif
  void SubmitFinalDisplayFrame(const FinalDisplayFrameView& frame) override;
  auto GetWidth() const -> int override;
  auto GetHeight() const -> int override;
  auto GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> override;
  void SetNextFramePresentationMode(FramePresentationMode mode) override;
  auto GetViewerSurface() -> IEditViewerSurface* override;
  auto GetViewerSurface() const -> const IEditViewerSurface* override;

 private:
  IFrameSink*                    downstream_sink_ = nullptr;
  std::shared_ptr<IScopeAnalyzer> scope_analyzer_ = {};
  mutable std::mutex             mutex_{};
  FinalDisplayFrameView          current_frame_{};
  ScopeRequest                   scope_request_{};
};

}  // namespace puerhlab
