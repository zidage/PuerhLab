//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.
#pragma once

#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#ifdef HAVE_METAL
#include <puerhlab/metal/Metal.hpp>

namespace puerhlab {
namespace metal {
using MetalTextureHandle = MTL::Texture*;

enum class PixelFormat {
  R16UINT,
  RGBA16UINT,
  R16FLOAT,
  RGBA16FLOAT,
  R32FLOAT,
  RGBA32FLOAT,
};

class MetalImage {
 private:
  uint32_t                    width_         = 0;
  uint32_t                    height_        = 0;
  uint32_t                    usage_flags_   = static_cast<uint32_t>(MTL::TextureUsageUnknown);
  PixelFormat                 format_        = PixelFormat::RGBA32FLOAT;
  NS::SharedPtr<MTL::Texture> texture_owner_ = nullptr;

  MetalImage(NS::SharedPtr<MTL::Texture>&& texture_owner, uint32_t width, uint32_t height,
             PixelFormat format, MTL::TextureUsage usage) noexcept;

 public:
  MetalImage()                                          = default;
  ~MetalImage()                                         = default;

  MetalImage(const MetalImage&) noexcept                  = default;
  auto operator=(const MetalImage&) noexcept -> MetalImage& = default;

  MetalImage(MetalImage&&) noexcept                      = default;
  auto operator=(MetalImage&&) noexcept -> MetalImage&  = default;

  static MetalImage Create2D(uint32_t width, uint32_t height, PixelFormat format,
                             bool shader_read = true, bool shader_write = true,
                             bool render_target = false);
  static MetalImage Wrap(MetalTextureHandle texture);
  static auto       PixelFormatFromCVType(int cv_type) -> PixelFormat;
  static auto       CVTypeFromPixelFormat(PixelFormat format) -> int;

  void              Create(uint32_t width, uint32_t height, PixelFormat format,
                           bool shader_read = true, bool shader_write = true,
                           bool render_target = false);
  void              Upload(const cv::Mat& host_image);
  void              Download(cv::Mat& host_image) const;
  void              CopyTo(MetalImage& dst) const;
  void              CropTo(MetalImage& dst, const cv::Rect& crop_rect) const;
  void              ConvertTo(MetalImage& dst, PixelFormat dst_format, double alpha = 1.0,
                              double beta = 0.0) const;
  void              Release() noexcept;
  void              Swap(MetalImage& other) noexcept;

  bool              Empty() const noexcept { return texture_owner_.get() == nullptr; }
  bool              IsValid() const noexcept { return !Empty(); }
  explicit operator bool() const noexcept { return IsValid(); }

  auto              Width() const noexcept { return width_; }
  auto              Height() const noexcept { return height_; }
  auto              Format() const noexcept { return format_; }
  auto              Usage() const noexcept -> MTL::TextureUsage {
    return static_cast<MTL::TextureUsage>(usage_flags_);
  }

  MTL::Texture* Texture() const noexcept { return texture_owner_.get(); }
};
}  // namespace metal
}  // namespace puerhlab
#endif
