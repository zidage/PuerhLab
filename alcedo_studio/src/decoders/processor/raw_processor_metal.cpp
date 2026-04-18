//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/raw_processor.hpp"

#ifdef HAVE_METAL

#include "decoders/processor/operators/gpu/metal_cvt_ref_space.hpp"
#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/metal_to_linear_ref.hpp"
#include "decoders/processor/operators/gpu/metal_xtrans_interpolate.hpp"
#include "decoders/processor/raw_processor_internal.hpp"
#include "metal/metal_utils/metal_convert_utils.hpp"

namespace alcedo {
namespace {

void ApplyMetalGeometricCorrections(metal::MetalImage& gpu_img, const int flip) {
  switch (flip) {
    case 3:
      metal::utils::Rotate180(gpu_img);
      break;
    case 5:
      metal::utils::Rotate90CCW(gpu_img);
      break;
    case 6:
      metal::utils::Rotate90CW(gpu_img);
      break;
    default:
      break;
  }
}

}  // namespace

auto RawProcessor::ProcessDirectRgbMetal() -> ImageBuffer {
  process_buffer_.SyncToGPU();
  process_buffer_.ReleaseCPUData();
  auto& gpu_img = process_buffer_.GetMetalImage();
  ApplyMetalGeometricCorrections(gpu_img, raw_data_.sizes.flip);
  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessMetal() -> ImageBuffer {
  if (input_kind_ == RawInputKind::DebayeredRgb) {
    return ProcessDirectRgbMetal();
  }

  SetDecodeRes();
  process_buffer_.SyncToGPU();
  process_buffer_.ReleaseCPUData();

  auto& gpu_img = process_buffer_.GetMetalImage();
  metal::ToLinearRef(gpu_img, raw_processor_, cfa_pattern_);

  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && params_.highlights_reconstruct_) {
    metal::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    const cv::Rect crop_rect =
        detail::BuildDecodeCropRect(raw_data_.sizes, cv::Size(gpu_img.Width(), gpu_img.Height()),
                                    params_.decode_res_);
    if (!detail::IsFullImageRect(crop_rect, cv::Size(gpu_img.Width(), gpu_img.Height()))) {
      metal::MetalImage cropped;
      gpu_img.CropTo(cropped, crop_rect);
      gpu_img = std::move(cropped);
    }
    metal::HighlightReconstruct(gpu_img, raw_processor_);
  } else {
    metal::utils::ClampTexture(gpu_img);
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      const int passes = params_.decode_res_ == DecodeRes::FULL ? 3 : 1;
      metal::XTransToRGB_Ref(gpu_img, cfa_pattern_.xtrans_pattern, passes);
    } else {
      metal::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    }
    const cv::Rect crop_rect =
        detail::BuildDecodeCropRect(raw_data_.sizes, cv::Size(gpu_img.Width(), gpu_img.Height()),
                                    params_.decode_res_);
    if (!detail::IsFullImageRect(crop_rect, cv::Size(gpu_img.Width(), gpu_img.Height()))) {
      metal::MetalImage cropped;
      gpu_img.CropTo(cropped, crop_rect);
      gpu_img = std::move(cropped);
    }
  }

  metal::ApplyInverseCamMul(gpu_img, raw_data_.color.cam_mul);
  runtime_color_context_.output_in_camera_space_ = true;
  ApplyMetalGeometricCorrections(gpu_img, raw_data_.sizes.flip);
  return {std::move(process_buffer_)};
}

}  // namespace alcedo

#endif
