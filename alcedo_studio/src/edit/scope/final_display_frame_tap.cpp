//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "edit/scope/final_display_frame_tap.hpp"

#include <utility>

namespace alcedo {

FinalDisplayFrameTapSink::FinalDisplayFrameTapSink(
    IFrameSink* downstream_sink, std::shared_ptr<IScopeAnalyzer> scope_analyzer)
    : downstream_sink_(downstream_sink), scope_analyzer_(std::move(scope_analyzer)) {}

void FinalDisplayFrameTapSink::SetScopeRequest(const ScopeRequest& request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    scope_request_ = request;
  }
  if (scope_analyzer_) {
    scope_analyzer_->ResizeResources(request);
  }
}

auto FinalDisplayFrameTapSink::GetScopeRequest() const -> ScopeRequest {
  std::lock_guard<std::mutex> lock(mutex_);
  return scope_request_;
}

auto FinalDisplayFrameTapSink::GetCurrentDisplayFrameView() const -> FinalDisplayFrameView {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_frame_;
}

void FinalDisplayFrameTapSink::EnsureSize(int width, int height) {
  if (downstream_sink_) {
    downstream_sink_->EnsureSize(width, height);
  }
}

auto FinalDisplayFrameTapSink::MapResourceForWrite() -> FrameWriteMapping {
  return downstream_sink_ ? downstream_sink_->MapResourceForWrite() : FrameWriteMapping{};
}

void FinalDisplayFrameTapSink::UnmapResource() {
  if (downstream_sink_) {
    downstream_sink_->UnmapResource();
  }
}

void FinalDisplayFrameTapSink::NotifyFrameReady() {
  if (downstream_sink_) {
    downstream_sink_->NotifyFrameReady();
  }
}

void FinalDisplayFrameTapSink::SubmitHostFrame(const ViewerFrame& frame) {
  if (downstream_sink_) {
    downstream_sink_->SubmitHostFrame(frame);
  }
}

#ifdef HAVE_METAL
void FinalDisplayFrameTapSink::SubmitMetalFrame(const ViewerMetalFrame& frame) {
  if (downstream_sink_) {
    downstream_sink_->SubmitMetalFrame(frame);
  }
}
#endif

void FinalDisplayFrameTapSink::SubmitFinalDisplayFrame(const FinalDisplayFrameView& frame) {
  ScopeRequest request;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_frame_ = frame;
    request        = scope_request_;
  }

  if (scope_analyzer_ && frame) {
    scope_analyzer_->SubmitFrame(frame, request);
  }
}

auto FinalDisplayFrameTapSink::GetWidth() const -> int {
  return downstream_sink_ ? downstream_sink_->GetWidth() : 0;
}

auto FinalDisplayFrameTapSink::GetHeight() const -> int {
  return downstream_sink_ ? downstream_sink_->GetHeight() : 0;
}

auto FinalDisplayFrameTapSink::GetViewportRenderRegion() const
    -> std::optional<ViewportRenderRegion> {
  return downstream_sink_ ? downstream_sink_->GetViewportRenderRegion() : std::nullopt;
}

void FinalDisplayFrameTapSink::SetNextFramePresentationMode(FramePresentationMode mode) {
  if (downstream_sink_) {
    downstream_sink_->SetNextFramePresentationMode(mode);
  }
}

auto FinalDisplayFrameTapSink::GetViewerSurface() -> IEditViewerSurface* {
  return downstream_sink_ ? downstream_sink_->GetViewerSurface() : nullptr;
}

auto FinalDisplayFrameTapSink::GetViewerSurface() const -> const IEditViewerSurface* {
  return downstream_sink_ ? downstream_sink_->GetViewerSurface() : nullptr;
}

}  // namespace alcedo
