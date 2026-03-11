//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>

#include "ui/edit_viewer/frame_sink.hpp"

namespace puerhlab {

class FrameMailbox {
 public:
  struct SlotDescriptor {
    int width  = 0;
    int height = 0;
  };

  struct ResizeDecision {
    bool need_resize = false;
    int  slot_index  = 0;
  };

  struct ActiveFrame {
    int                   slot_index         = 0;
    int                   width              = 0;
    int                   height             = 0;
    FramePresentationMode presentation_mode  = FramePresentationMode::FullFrame;
  };

  struct PendingFrame {
    int                   slot_index                = -1;
    int                   width                     = 0;
    int                   height                    = 0;
    void*                 staging_ptr               = nullptr;
    size_t                staging_bytes             = 0;
    bool                  apply_presentation_mode   = false;
    FramePresentationMode presentation_mode         = FramePresentationMode::FullFrame;
  };

  FrameMailbox();
  ~FrameMailbox();

  void InitializeDefaultSize(int width, int height);
  auto EnsureSize(int width, int height) -> ResizeDecision;
  void CommitResize(int slot_index, int width, int height);

  auto MapResourceForWrite() -> FrameWriteMapping;
  void UnmapResource();
  auto ConsumePendingFrame() -> std::optional<PendingFrame>;
  void MarkFramePresented(int slot_index, bool apply_presentation_mode);

  auto GetActiveFrame() const -> ActiveFrame;
  auto GetWidth() const -> int;
  auto GetHeight() const -> int;
  auto GetRenderTargetSlotDescriptor() const -> SlotDescriptor;
  auto GetRenderTargetSlotIndex() const -> int;
  auto GetSlotDescriptor(int slot_index) const -> SlotDescriptor;

  void SetNextFramePresentationMode(FramePresentationMode mode);
  void SetHistogramFrameExpected(bool expected_fast_preview);
  void NotifyFrameReady();
  auto ConsumeHistogramPendingFrame() -> bool;

 private:
  void EnsureStagingCapacityLocked(size_t needed_bytes);

  std::array<SlotDescriptor, 2>         slots_{};
  int                                   active_idx_        = 0;
  int                                   write_idx_         = 1;
  int                                   render_target_idx_ = 0;
  void*                                 staging_ptr_       = nullptr;
  size_t                                staging_bytes_     = 0;
  mutable std::mutex                    mutex_;
  std::atomic<int>                      pending_frame_idx_{-1};
  std::atomic<FramePresentationMode>    active_presentation_mode_{
      FramePresentationMode::FullFrame};
  std::atomic<FramePresentationMode>    pending_presentation_mode_{
      FramePresentationMode::FullFrame};
  std::atomic<bool>                     pending_presentation_mode_valid_{false};
  std::atomic<bool>                     histogram_expect_fast_frame_{false};
  std::atomic<bool>                     histogram_pending_frame_{false};
};

}  // namespace puerhlab
