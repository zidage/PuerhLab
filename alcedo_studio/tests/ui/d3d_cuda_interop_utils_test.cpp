//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#if defined(_WIN32) && defined(HAVE_CUDA)
#include <array>
#include <cstdint>

#include "ui/edit_viewer/d3d_cuda_interop_utils.hpp"

namespace alcedo {
namespace {

TEST(D3DCudaInteropUtilsTest, DedicatedExternalMemoryDescCarriesD3D12SharedHandle) {
  auto* fake_shared_handle = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(0x1234));

  const cudaExternalMemoryHandleDesc desc = MakeDedicatedCudaExternalMemoryHandleDesc(
      fake_shared_handle, cudaExternalMemoryHandleTypeD3D12Resource, 4096ULL);

  EXPECT_EQ(desc.type, cudaExternalMemoryHandleTypeD3D12Resource);
  EXPECT_EQ(desc.handle.win32.handle, fake_shared_handle);
  EXPECT_EQ(desc.size, 4096ULL);
  EXPECT_EQ(desc.flags, cudaExternalMemoryDedicated);
}

TEST(D3DCudaInteropUtilsTest, DedicatedExternalMemoryDescCarriesD3D11SharedHandle) {
  auto* fake_shared_handle = reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(0x5678));

  const cudaExternalMemoryHandleDesc desc = MakeDedicatedCudaExternalMemoryHandleDesc(
      fake_shared_handle, cudaExternalMemoryHandleTypeD3D11Resource, 8192ULL);

  EXPECT_EQ(desc.type, cudaExternalMemoryHandleTypeD3D11Resource);
  EXPECT_EQ(desc.handle.win32.handle, fake_shared_handle);
  EXPECT_EQ(desc.size, 8192ULL);
  EXPECT_EQ(desc.flags, cudaExternalMemoryDedicated);
}

TEST(D3DCudaInteropUtilsTest, WriteSlotSelectionAvoidsActiveAndPendingSlots) {
  std::array<DirectPresentSlotAvailability, 3> slots = {
      DirectPresentSlotAvailability{1920, 1080, true, true},
      DirectPresentSlotAvailability{1920, 1080, true, true},
      DirectPresentSlotAvailability{1920, 1080, true, false},
  };

  const DirectPresentTargetSelection selection =
      SelectDirectPresentWriteSlot(slots.data(), slots.size(), 1, 1920, 1080);

  EXPECT_EQ(selection.slot_index, 2);
  EXPECT_FALSE(selection.need_resize);
}

TEST(D3DCudaInteropUtilsTest, WriteSlotSelectionPrefersReusableFreeSlotOverResize) {
  std::array<DirectPresentSlotAvailability, 3> slots = {
      DirectPresentSlotAvailability{1920, 1080, true, true},
      DirectPresentSlotAvailability{1280, 720, true, false},
      DirectPresentSlotAvailability{1920, 1080, true, false},
  };

  const DirectPresentTargetSelection selection =
      SelectDirectPresentWriteSlot(slots.data(), slots.size(), 1, 1920, 1080);

  EXPECT_EQ(selection.slot_index, 2);
  EXPECT_FALSE(selection.need_resize);

  slots[1].unavailable = true;
  const DirectPresentTargetSelection reusable_selection =
      SelectDirectPresentWriteSlot(slots.data(), slots.size(), 1, 1920, 1080);

  EXPECT_EQ(reusable_selection.slot_index, 2);
  EXPECT_FALSE(reusable_selection.need_resize);
}

}  // namespace
}  // namespace alcedo
#endif
