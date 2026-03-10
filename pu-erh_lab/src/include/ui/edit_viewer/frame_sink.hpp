//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstddef>

#include <optional>

namespace puerhlab {
struct ViewportRenderRegion {
  int   x_            = 0;
  int   y_            = 0;
  float scale_x_      = 1.0f;
  float scale_y_      = 1.0f;
};

enum class FramePresentationMode {
  ViewportTransformed,
  RoiFrame,
};

enum class FramePixelFormat {
  RGBA32F,
};

enum class FrameMemoryDomain {
  HostVisible,
  CudaDevice,
};

struct FrameWriteMapping {
  void*             data          = nullptr;
  size_t            row_bytes     = 0;
  FramePixelFormat  pixel_format  = FramePixelFormat::RGBA32F;
  FrameMemoryDomain memory_domain = FrameMemoryDomain::HostVisible;

  explicit operator bool() const { return data != nullptr; }
};

class IFrameSink {
 public:
  virtual ~IFrameSink() {}

  virtual void    EnsureSize(int width, int height) = 0;

  virtual auto    MapResourceForWrite() -> FrameWriteMapping = 0;

  virtual void    UnmapResource()                   = 0;

  virtual void    NotifyFrameReady()                = 0;

  // Get the size of the frame
  virtual int     GetWidth() const                  = 0;
  virtual int     GetHeight() const                 = 0;

  // Returns ROI parameters derived from the current viewer transform (if any).
  virtual auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
    return std::nullopt;
  }

  // Sets how the next presented frame should be displayed.
  virtual void    SetNextFramePresentationMode(FramePresentationMode) {}
};
}  // namespace puerhlab
