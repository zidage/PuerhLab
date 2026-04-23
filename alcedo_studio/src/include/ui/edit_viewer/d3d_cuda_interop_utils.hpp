//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#if defined(_WIN32) && defined(HAVE_CUDA)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace alcedo {

struct DirectPresentSlotAvailability {
  int  width          = 0;
  int  height         = 0;
  bool has_resource   = false;
  bool unavailable    = false;
};

struct DirectPresentTargetSelection {
  int  slot_index  = 0;
  bool need_resize = false;
};

inline auto MakeDedicatedCudaExternalMemoryHandleDesc(
    HANDLE shared_handle, cudaExternalMemoryHandleType handle_type, unsigned long long size_bytes)
    -> cudaExternalMemoryHandleDesc {
  cudaExternalMemoryHandleDesc handle_desc{};
  handle_desc.type                = handle_type;
  handle_desc.handle.win32.handle = shared_handle;
  handle_desc.size                = size_bytes;
  handle_desc.flags               = cudaExternalMemoryDedicated;
  return handle_desc;
}

inline auto IsReusableDirectPresentSlot(const DirectPresentSlotAvailability& slot, int width,
                                        int height) -> bool {
  return !slot.unavailable && slot.has_resource && slot.width == width && slot.height == height;
}

inline auto SelectDirectPresentWriteSlot(const DirectPresentSlotAvailability* slot_infos,
                                         std::size_t slot_count, int preferred_slot, int width,
                                         int height) -> DirectPresentTargetSelection {
  if (!slot_infos || slot_count == 0) {
    return {};
  }

  auto is_valid_index = [slot_count](int slot_index) {
    return slot_index >= 0 && static_cast<std::size_t>(slot_index) < slot_count;
  };

  int selected_slot = is_valid_index(preferred_slot) ? preferred_slot : 0;
  if (IsReusableDirectPresentSlot(slot_infos[selected_slot], width, height)) {
    return {selected_slot, false};
  }

  for (std::size_t i = 0; i < slot_count; ++i) {
    if (IsReusableDirectPresentSlot(slot_infos[i], width, height)) {
      return {static_cast<int>(i), false};
    }
  }

  if (!slot_infos[selected_slot].unavailable) {
    return {selected_slot, true};
  }

  for (std::size_t i = 0; i < slot_count; ++i) {
    if (!slot_infos[i].unavailable) {
      selected_slot = static_cast<int>(i);
      return {selected_slot, true};
    }
  }

  return {selected_slot, !IsReusableDirectPresentSlot(slot_infos[selected_slot], width, height)};
}

}  // namespace alcedo
#endif
