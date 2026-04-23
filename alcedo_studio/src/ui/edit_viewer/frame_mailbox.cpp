//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/edit_viewer/frame_mailbox.hpp"

#include <algorithm>

#include <QDebug>

#include <cuda_runtime_api.h>

namespace alcedo {

FrameMailbox::FrameMailbox() = default;

FrameMailbox::~FrameMailbox() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (staging_ptr_) {
    cudaFree(staging_ptr_);
    staging_ptr_   = nullptr;
    staging_bytes_ = 0;
  }
}

void FrameMailbox::InitializeDefaultSize(int width, int height) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (slots_[active_idx_].width <= 0 || slots_[active_idx_].height <= 0) {
    slots_[active_idx_].width  = std::max(1, width);
    slots_[active_idx_].height = std::max(1, height);
  }
}

auto FrameMailbox::EnsureSize(int width, int height) -> ResizeDecision {
  ResizeDecision decision{};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& target_slot = slots_[render_target_idx_];
    if (target_slot.width != width || target_slot.height != height) {
      render_target_idx_ = write_idx_;
      const auto& write_slot = slots_[render_target_idx_];
      decision.need_resize = (write_slot.width != width || write_slot.height != height);
      decision.slot_index  = render_target_idx_;
    } else {
      decision.slot_index = render_target_idx_;
    }

    const size_t needed_bytes =
        static_cast<size_t>(std::max(width, 0)) * static_cast<size_t>(std::max(height, 0)) *
        sizeof(float4);
    EnsureStagingCapacityLocked(needed_bytes);
  }
  return decision;
}

void FrameMailbox::CommitResize(int slot_index, int width, int height) {
  if (slot_index < 0 || slot_index >= static_cast<int>(slots_.size())) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  slots_[slot_index].width  = width;
  slots_[slot_index].height = height;
  render_target_idx_        = slot_index;
}

auto FrameMailbox::MapResourceForWrite() -> FrameWriteMapping {
  mutex_.lock();
  if (!staging_ptr_ || staging_bytes_ == 0) {
    mutex_.unlock();
    return {};
  }

  const auto& slot = slots_[render_target_idx_];
  return {staging_ptr_,
          nullptr,
          static_cast<size_t>(std::max(slot.width, 0)) * sizeof(float4),
          FramePixelFormat::RGBA32F,
          FrameMemoryDomain::CudaDevice,
          FrameWriteTargetType::LinearBuffer,
          0,
          nullptr,
          0};
}

void FrameMailbox::UnmapResource() {
  pending_frame_idx_.store(render_target_idx_, std::memory_order_release);
  mutex_.unlock();
}

auto FrameMailbox::ConsumePendingFrame() -> std::optional<PendingFrame> {
  const int pending_slot = pending_frame_idx_.exchange(-1, std::memory_order_acq_rel);
  if (pending_slot < 0 || pending_slot >= static_cast<int>(slots_.size())) {
    return std::nullopt;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  PendingFrame pending{};
  pending.slot_index              = pending_slot;
  pending.width                   = slots_[pending_slot].width;
  pending.height                  = slots_[pending_slot].height;
  pending.staging_ptr             = staging_ptr_;
  pending.staging_bytes           = staging_bytes_;
  pending.apply_presentation_mode =
      pending_presentation_mode_valid_.load(std::memory_order_acquire);
  pending.presentation_mode = pending_presentation_mode_.load(std::memory_order_acquire);
  return pending;
}

void FrameMailbox::MarkFramePresented(int slot_index, bool apply_presentation_mode) {
  if (slot_index < 0 || slot_index >= static_cast<int>(slots_.size())) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  active_idx_ = slot_index;
  write_idx_  = 1 - active_idx_;
  if (apply_presentation_mode &&
      pending_presentation_mode_valid_.exchange(false, std::memory_order_acq_rel)) {
    active_presentation_mode_.store(pending_presentation_mode_.load(std::memory_order_acquire),
                                    std::memory_order_release);
  }
}

auto FrameMailbox::GetActiveFrame() const -> ActiveFrame {
  std::lock_guard<std::mutex> lock(mutex_);
  return {active_idx_, slots_[active_idx_].width, slots_[active_idx_].height,
          active_presentation_mode_.load(std::memory_order_acquire)};
}

auto FrameMailbox::GetWidth() const -> int {
  std::lock_guard<std::mutex> lock(mutex_);
  return slots_[active_idx_].width;
}

auto FrameMailbox::GetHeight() const -> int {
  std::lock_guard<std::mutex> lock(mutex_);
  return slots_[active_idx_].height;
}

auto FrameMailbox::GetRenderTargetSlotDescriptor() const -> SlotDescriptor {
  std::lock_guard<std::mutex> lock(mutex_);
  return slots_[render_target_idx_];
}

auto FrameMailbox::GetRenderTargetSlotIndex() const -> int {
  std::lock_guard<std::mutex> lock(mutex_);
  return render_target_idx_;
}

auto FrameMailbox::GetSlotDescriptor(int slot_index) const -> SlotDescriptor {
  if (slot_index < 0 || slot_index >= static_cast<int>(slots_.size())) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  return slots_[slot_index];
}

void FrameMailbox::SetNextFramePresentationMode(FramePresentationMode mode) {
  pending_presentation_mode_.store(mode, std::memory_order_release);
  pending_presentation_mode_valid_.store(true, std::memory_order_release);
}

void FrameMailbox::SetHistogramFrameExpected(bool expected_fast_preview) {
  histogram_expect_fast_frame_.store(expected_fast_preview, std::memory_order_release);
}

void FrameMailbox::NotifyFrameReady() {
  histogram_pending_frame_.store(histogram_expect_fast_frame_.load(std::memory_order_acquire),
                                 std::memory_order_release);
}

auto FrameMailbox::ConsumeHistogramPendingFrame() -> bool {
  return histogram_pending_frame_.exchange(false, std::memory_order_acq_rel);
}

void FrameMailbox::EnsureStagingCapacityLocked(size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= staging_bytes_) {
    return;
  }

  void* new_staging_ptr = nullptr;
  const cudaError_t alloc_err = cudaMalloc(&new_staging_ptr, needed_bytes);
  if (alloc_err != cudaSuccess) {
    qWarning("Failed to allocate CUDA staging buffer (%zu bytes): %s", needed_bytes,
             cudaGetErrorString(alloc_err));
    return;
  }

  if (staging_ptr_) {
    cudaFree(staging_ptr_);
  }
  staging_ptr_   = new_staging_ptr;
  staging_bytes_ = needed_bytes;
}

}  // namespace alcedo
