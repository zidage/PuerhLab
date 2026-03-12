//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <optional>

#include "edit/operators/utils/color_utils.hpp"

namespace puerhlab {
class IEditViewerSurface;

struct ViewerDisplayConfig {
  ColorUtils::ColorSpace encoding_space = ColorUtils::ColorSpace::REC709;
  ColorUtils::EOTF       encoding_eotf  = ColorUtils::EOTF::GAMMA_2_2;

  auto operator==(const ViewerDisplayConfig& other) const -> bool = default;
};

struct ViewportRenderRegion {
  int   x_            = 0;
  int   y_            = 0;
  float scale_x_      = 1.0f;
  float scale_y_      = 1.0f;
};

enum class FramePresentationMode {
  FullFrame,
  ViewportTransformed = FullFrame,
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

struct ViewerFrame {
  int                   width               = 0;
  int                   height              = 0;
  size_t                row_bytes           = 0;
  std::shared_ptr<const void> pixels{};
  ViewerDisplayConfig   display_config{};
  FramePresentationMode presentation_mode   = FramePresentationMode::FullFrame;

  explicit operator bool() const {
    return width > 0 && height > 0 && row_bytes > 0 && pixels != nullptr;
  }
};

struct ViewerGpuFrameUpload {
  int                   width               = 0;
  int                   height              = 0;
  size_t                row_bytes           = 0;
  std::shared_ptr<const void> pixels{};
  ViewerDisplayConfig   display_config{};
  FramePresentationMode presentation_mode   = FramePresentationMode::FullFrame;

  explicit operator bool() const {
    return width > 0 && height > 0 && row_bytes > 0 && pixels != nullptr;
  }
};

#ifdef HAVE_METAL
struct ViewerMetalFrame {
  int                   width               = 0;
  int                   height              = 0;
  std::uintptr_t        texture_handle      = 0;
  std::shared_ptr<const void> owner{};
  ViewerDisplayConfig   display_config{};
  FramePresentationMode presentation_mode   = FramePresentationMode::FullFrame;

  explicit operator bool() const {
    return width > 0 && height > 0 && texture_handle != 0 && owner != nullptr;
  }
};
#endif

class IFrameSink {
 public:
  virtual ~IFrameSink() {}

  virtual void    EnsureSize(int width, int height) = 0;

  virtual auto    MapResourceForWrite() -> FrameWriteMapping = 0;

  virtual void    UnmapResource()                   = 0;

  virtual void    NotifyFrameReady()                = 0;

  virtual void    SubmitHostFrame(const ViewerFrame&) {}
#ifdef HAVE_METAL
  virtual void    SubmitMetalFrame(const ViewerMetalFrame&) {}
#endif

  // Get the size of the frame
  virtual int     GetWidth() const                  = 0;
  virtual int     GetHeight() const                 = 0;

  // Returns ROI parameters derived from the current viewer transform (if any).
  virtual auto    GetViewportRenderRegion() const -> std::optional<ViewportRenderRegion> {
    return std::nullopt;
  }

  // Sets how the next presented frame should be displayed.
  virtual void    SetNextFramePresentationMode(FramePresentationMode) {}

  // Exposes the presentation surface when a sink is backed by a live viewer.
  virtual auto    GetViewerSurface() -> IEditViewerSurface* { return nullptr; }
  virtual auto    GetViewerSurface() const -> const IEditViewerSurface* { return nullptr; }
};
}  // namespace puerhlab
