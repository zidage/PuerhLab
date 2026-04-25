//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/raw_processor.hpp"

#ifdef HAVE_WEBGPU

#include <stdexcept>
#include <string>

#include "decoders/processor/operators/gpu/webgpu_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/webgpu_to_linear_ref.hpp"
#include "decoders/processor/raw_processor_internal.hpp"
#include "image/gpu_backend.hpp"
#include "webgpu/webgpu_context.hpp"
#include "webgpu/webgpu_geometry_utils.hpp"

namespace alcedo {
namespace {

void EnsureWebGpuRawBackendAvailable() {
  auto& context = webgpu::WebGpuContext::Instance();
  if (context.IsAvailable()) {
    return;
  }

  std::string message = "RawProcessor: WebGPU backend is unavailable.";
  if (!context.InitializationLog().empty()) {
    message += " Dawn initialization log: " + context.InitializationLog();
  }
  throw std::runtime_error(message);
}

auto IsUnorientedOrIdentityFlip(const int flip) -> bool { return flip == 0 || flip == 1; }

void ApplyWebGpuGeometricCorrections(webgpu::WebGpuImage& gpu_img, const int flip) {
  switch (flip) {
    case 3:
      webgpu::utils::Rotate180(gpu_img);
      break;
    case 5:
      webgpu::utils::Rotate90CCW(gpu_img);
      break;
    case 6:
      webgpu::utils::Rotate90CW(gpu_img);
      break;
    default:
      break;
  }
}

}  // namespace

auto RawProcessor::ProcessDirectRgbWebGpu() -> ImageBuffer {
  EnsureWebGpuRawBackendAvailable();
  process_buffer_.SyncToGPU(GpuBackendKind::WebGPU);
  process_buffer_.ReleaseCPUData();

  if (!IsUnorientedOrIdentityFlip(raw_data_.sizes.flip)) {
    auto& gpu_img = process_buffer_.GetWebGpuImage();
    ApplyWebGpuGeometricCorrections(gpu_img, raw_data_.sizes.flip);
  }

  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessWebGpu() -> ImageBuffer {
  EnsureWebGpuRawBackendAvailable();
  if (input_kind_ == RawInputKind::DebayeredRgb) {
    return ProcessDirectRgbWebGpu();
  }

  SetDecodeRes();
  process_buffer_.SyncToGPU(GpuBackendKind::WebGPU);
  process_buffer_.ReleaseCPUData();

  auto& gpu_img = process_buffer_.GetWebGpuImage();
  webgpu::ToLinearRef(gpu_img, raw_processor_, cfa_pattern_);

  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && params_.highlights_reconstruct_) {
    webgpu::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    const cv::Rect crop_rect = detail::BuildDecodeCropRect(
        raw_data_.sizes, cv::Size(gpu_img.Width(), gpu_img.Height()), params_.decode_res_);
    if (!detail::IsFullImageRect(crop_rect, cv::Size(gpu_img.Width(), gpu_img.Height()))) {
      cv::Mat host;
      gpu_img.Download(host);
      cv::Mat cropped = host(crop_rect).clone();
      gpu_img.Upload(cropped);
    }
    throw std::runtime_error(
        "RawProcessor: WebGPU highlight reconstruction is not implemented yet.");
  } else {
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      throw std::runtime_error(
          "RawProcessor: WebGPU X-Trans interpolation is not implemented yet.");
    } else {
      webgpu::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    }
    const cv::Rect crop_rect = detail::BuildDecodeCropRect(
        raw_data_.sizes, cv::Size(gpu_img.Width(), gpu_img.Height()), params_.decode_res_);
    if (!detail::IsFullImageRect(crop_rect, cv::Size(gpu_img.Width(), gpu_img.Height()))) {
      cv::Mat host;
      gpu_img.Download(host);
      cv::Mat cropped = host(crop_rect).clone();
      gpu_img.Upload(cropped);
    }
  }

  ApplyWebGpuGeometricCorrections(gpu_img, raw_data_.sizes.flip);
  return {std::move(process_buffer_)};
}

}  // namespace alcedo

#endif
