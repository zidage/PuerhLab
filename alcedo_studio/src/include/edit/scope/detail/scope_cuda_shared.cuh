//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "ui/edit_viewer/frame_sink.hpp"

namespace alcedo::scope::cuda_detail {

struct CudaLinearImageResource {
  void*             device_ptr  = nullptr;
  size_t            row_bytes   = 0;
  int               width       = 0;
  int               height      = 0;
  FramePixelFormat  format      = FramePixelFormat::RGBA32F;
  bool              owns_memory = false;
  std::uintptr_t    native_object = 0;

  ~CudaLinearImageResource() {
    if (owns_memory && device_ptr) {
      cudaFree(device_ptr);
      device_ptr = nullptr;
    }
  }
};

struct CudaDeviceBufferResource {
  void*   device_ptr  = nullptr;
  size_t  size_bytes  = 0;
  bool    owns_memory = false;

  ~CudaDeviceBufferResource() {
    if (owns_memory && device_ptr) {
      cudaFree(device_ptr);
      device_ptr = nullptr;
    }
  }
};

struct CudaStreamSignalResource {
  cudaStream_t stream = nullptr;
};

}  // namespace alcedo::scope::cuda_detail
