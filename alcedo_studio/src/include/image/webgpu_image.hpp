//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.
#pragma once

#ifdef HAVE_WEBGPU

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <stdexcept>

namespace alcedo {
namespace webgpu {

enum class PixelFormat {
  R16UINT,
  RGBA16UINT,
  R16FLOAT,
  RGBA16FLOAT,
  R32FLOAT,
  RGBA32FLOAT,
};

class WebGpuImage {
 public:
  WebGpuImage()                                                 = default;
  ~WebGpuImage()                                                = default;
  WebGpuImage(const WebGpuImage&)                               = default;
  auto operator=(const WebGpuImage&) -> WebGpuImage&            = default;
  WebGpuImage(WebGpuImage&&) noexcept                           = default;
  auto        operator=(WebGpuImage&&) noexcept -> WebGpuImage& = default;

  static auto PixelFormatFromCVType(int cv_type) -> PixelFormat;
  static auto CVTypeFromPixelFormat(PixelFormat format) -> int;

  void Create(uint32_t width, uint32_t height, PixelFormat format, bool texture_binding = true,
              bool storage_binding = true);
  void Upload(const cv::Mat& host_image);
  void Download(cv::Mat& host_image) const;
  void CopyTo(WebGpuImage& dst) const;
  void ConvertTo(WebGpuImage& dst, PixelFormat dst_format, double alpha = 1.0,
                 double beta = 0.0) const;
  void Release() noexcept;

  [[nodiscard]] auto Empty() const noexcept -> bool { return texture_.Get() == nullptr; }
  [[nodiscard]] auto IsValid() const noexcept -> bool { return !Empty(); }
  explicit           operator bool() const noexcept { return IsValid(); }

  [[nodiscard]] auto Width() const noexcept { return width_; }
  [[nodiscard]] auto Height() const noexcept { return height_; }
  [[nodiscard]] auto Format() const noexcept { return format_; }
  [[nodiscard]] auto Texture() const noexcept -> const wgpu::Texture& { return texture_; }

 private:
  uint32_t      width_   = 0;
  uint32_t      height_  = 0;
  PixelFormat   format_  = PixelFormat::RGBA32FLOAT;
  wgpu::Texture texture_ = nullptr;
};

}  // namespace webgpu
}  // namespace alcedo

#endif
