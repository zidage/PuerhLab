//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.
#pragma once

#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <stdexcept>

#ifdef HAVE_METAL
#include <puerhlab/metal/Metal.hpp>

#include "metal/metal_utils/metal_convert_utils.hpp"

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

  static auto       Create2D(uint32_t width, uint32_t height, PixelFormat format,
                             bool shader_read = true, bool shader_write = true,
                             bool render_target = false) -> MetalImage {
    MetalImage image;
    image.Create(width, height, format, shader_read, shader_write, render_target);
    return image;
  }
  static MetalImage Wrap(MetalTextureHandle texture);
  static auto       PixelFormatFromCVType(int cv_type) -> PixelFormat {
    switch (cv_type) {
      case CV_16UC1:
        return PixelFormat::R16UINT;
      case CV_16UC4:
        return PixelFormat::RGBA16UINT;
      case CV_16FC1:
        return PixelFormat::R16FLOAT;
      case CV_16FC4:
        return PixelFormat::RGBA16FLOAT;
      case CV_32FC1:
        return PixelFormat::R32FLOAT;
      case CV_32FC4:
        return PixelFormat::RGBA32FLOAT;
      default:
        throw std::invalid_argument("MetalImage: Unsupported OpenCV type for Metal texture.");
    }
  }
  static auto       CVTypeFromPixelFormat(PixelFormat format) -> int {
    switch (format) {
      case PixelFormat::R16UINT:
        return CV_16UC1;
      case PixelFormat::RGBA16UINT:
        return CV_16UC4;
      case PixelFormat::R16FLOAT:
        return CV_16FC1;
      case PixelFormat::RGBA16FLOAT:
        return CV_16FC4;
      case PixelFormat::R32FLOAT:
        return CV_32FC1;
      case PixelFormat::RGBA32FLOAT:
        return CV_32FC4;
    }

    throw std::invalid_argument("MetalImage: Unsupported Metal pixel format.");
  }

  void              Create(uint32_t width, uint32_t height, PixelFormat format,
                           bool shader_read = true, bool shader_write = true,
                           bool render_target = false);
  void              Upload(const cv::Mat& host_image);
  void              Download(cv::Mat& host_image) const;
  void              CopyTo(MetalImage& dst) const;
  void              CropTo(MetalImage& dst, const cv::Rect& crop_rect) const {
    if (Empty()) {
      throw std::runtime_error("MetalImage: Cannot crop an empty texture.");
    }

    const bool preserve_render_target =
        (static_cast<uint32_t>(Usage()) & static_cast<uint32_t>(MTL::TextureUsageRenderTarget)) !=
        0U;
    dst.Create(static_cast<uint32_t>(crop_rect.width), static_cast<uint32_t>(crop_rect.height),
               format_, true, true, preserve_render_target);
    utils::CropTexture(*this, dst, crop_rect);
  }
  void              ConvertTo(MetalImage& dst, PixelFormat dst_format, double alpha = 1.0,
                              double beta = 0.0) const {
    if (Empty()) {
      throw std::runtime_error("MetalImage: Cannot convert an empty texture.");
    }

    if (format_ == dst_format && alpha == 1.0 && beta == 0.0) {
      CopyTo(dst);
      return;
    }

    const bool preserve_render_target =
        (static_cast<uint32_t>(Usage()) & static_cast<uint32_t>(MTL::TextureUsageRenderTarget)) !=
        0U;
    dst.Create(width_, height_, dst_format, true, true, preserve_render_target);
    utils::ConvertTexture(*this, dst, alpha, beta);
  }
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
