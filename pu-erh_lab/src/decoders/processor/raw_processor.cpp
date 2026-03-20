//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// TODO: Migrate to static pipeline architecture

#include "decoders/processor/raw_processor.hpp"

#include <libraw/libraw.h>  // Add this header for libraw_rawdata_t
#include <libraw/libraw_const.h>
#include <opencv2/core/hal/interface.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <string>

#include "decoders/processor/operators/cpu/debayer_rcd.hpp"
#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"
#include "type/type.hpp"

#ifdef HAVE_CUDA
#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"
#endif

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "decoders/processor/operators/cpu/color_space_conv.hpp"
#include "decoders/processor/operators/cpu/debayer_ahd.hpp"
#include "decoders/processor/operators/cpu/debayer_amaze.hpp"
#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/cpu/white_balance.hpp"

#ifdef HAVE_CUDA
#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_ahd.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/cuda_rotate.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"
#include "decoders/processor/operators/gpu/cuda_xtrans_interpolate.hpp"
#endif
#ifdef HAVE_METAL
#include "decoders/processor/operators/gpu/metal_cvt_ref_space.hpp"
#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/metal_to_linear_ref.hpp"
#include "decoders/processor/operators/gpu/metal_xtrans_interpolate.hpp"
#include "metal/metal_utils/metal_convert_utils.hpp"
#endif
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {
void ThrowUnsupportedGPUBackend(const char* op_name) {
#ifdef HAVE_METAL
  throw std::runtime_error(std::string("RawProcessor: ") + op_name +
                           " is not implemented for the Metal GPU backend.");
#else
  throw std::runtime_error(std::string("RawProcessor: ") + op_name +
                           " requires a compiled GPU backend implementation.");
#endif
}

auto BuildActiveAreaRect(const libraw_image_sizes_t& sizes, const cv::Size& image_size) -> cv::Rect {
  const int left   = std::clamp(static_cast<int>(sizes.left_margin), 0, image_size.width);
  const int top    = std::clamp(static_cast<int>(sizes.top_margin), 0, image_size.height);
  const int width  = std::clamp(static_cast<int>(sizes.width), 0, image_size.width - left);
  const int height = std::clamp(static_cast<int>(sizes.height), 0, image_size.height - top);

  if (width <= 0 || height <= 0) {
    return {0, 0, image_size.width, image_size.height};
  }
  return {left, top, width, height};
}

auto CropToActiveArea(const cv::Mat& src, const libraw_image_sizes_t& sizes) -> cv::Mat {
  const cv::Rect active_rect = BuildActiveAreaRect(sizes, src.size());
  return src(active_rect).clone();
}

auto BuildDirectRgbRgba(const libraw_rawdata_t& raw_data) -> cv::Mat {
  const auto& sizes     = raw_data.sizes;
  const int   raw_width = static_cast<int>(sizes.raw_width);
  const int   raw_height = static_cast<int>(sizes.raw_height);

  if (raw_data.color3_image != nullptr) {
    const size_t row_step = sizes.raw_pitch != 0 ? static_cast<size_t>(sizes.raw_pitch)
                                                 : static_cast<size_t>(raw_width) * sizeof(uint16_t) * 3;
    cv::Mat view(raw_height, raw_width, CV_16UC3, raw_data.color3_image, row_step);
    cv::Mat rgb32f;
    CropToActiveArea(view, sizes).convertTo(rgb32f, CV_32FC3, 1.0 / 65535.0);
    cv::Mat rgba32f;
    cv::cvtColor(rgb32f, rgba32f, cv::COLOR_RGB2RGBA);
    return rgba32f;
  }

  if (raw_data.float3_image != nullptr) {
    const size_t row_step = sizes.raw_pitch != 0 ? static_cast<size_t>(sizes.raw_pitch)
                                                 : static_cast<size_t>(raw_width) * sizeof(float) * 3;
    cv::Mat view(raw_height, raw_width, CV_32FC3, raw_data.float3_image, row_step);
    cv::Mat rgba32f;
    cv::cvtColor(CropToActiveArea(view, sizes), rgba32f, cv::COLOR_RGB2RGBA);
    return rgba32f;
  }

  throw std::runtime_error("RawProcessor: direct RGB input is missing a 3-channel source buffer.");
}

auto DescribeUnsupportedRawInput(LibRaw& raw_processor, const RawInputKind input_kind) -> std::string {
  const auto& idata   = raw_processor.imgdata.idata;
  const auto& rawdata = raw_processor.imgdata.rawdata;
  if (idata.is_foveon != 0U) {
    return "Foveon/X3F input is not supported by the pu-erh raw pipeline.";
  }
  if (raw_processor.is_fuji_rotated() != 0) {
    return "Fuji rotated CFA layouts are not supported.";
  }
  if (rawdata.color4_image != nullptr || rawdata.float4_image != nullptr) {
    return "4-channel decoded raw input is not supported.";
  }
  if (rawdata.float_image != nullptr) {
    return "single-channel float raw input is not supported.";
  }
  if (idata.filters == 0U) {
    return "the file does not expose a classic Bayer CFA.";
  }
  if (idata.filters == 1U) {
    return "non-2x2 tiled CFA layouts are not supported.";
  }
  if (idata.filters == 9U) {
    return "X-Trans CFA layouts are not supported.";
  }
  if (input_kind == RawInputKind::Unsupported) {
    return "no supported raw image plane was provided by LibRaw.";
  }
  return "unsupported raw input layout.";
}
}  // namespace

RawProcessor::RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                           LibRaw& raw_processor, const RawRuntimeColorContext& pre_ctx)
    : params_(params),
      runtime_color_context_(pre_ctx),
      raw_data_(rawdata),
      raw_processor_(raw_processor) {}

void RawProcessor::SetDecodeRes() {
  // Auto-downscale: if the longest side exceeds 8500px, force 1/8th resolution
  auto& cpu_data  = process_buffer_.GetCPUData();
  int   long_side = std::max(cpu_data.rows, cpu_data.cols);
  if (long_side > 8500 && params_.decode_res_ == DecodeRes::QUARTER) {
    params_.decode_res_ = DecodeRes::EIGHTH;
  }

  // Adjust internal parameters based on decode resolution
  switch (params_.decode_res_) {
    case DecodeRes::FULL:
      break;
    case DecodeRes::HALF:
      cpu_data = DownsampleRaw2x(cpu_data, cfa_pattern_);
      break;
    case DecodeRes::QUARTER:
      cpu_data = DownsampleRaw2x(DownsampleRaw2x(cpu_data, cfa_pattern_), cfa_pattern_);
      break;
    case DecodeRes::EIGHTH:
      cpu_data = DownsampleRaw2x(
          DownsampleRaw2x(DownsampleRaw2x(cpu_data, cfa_pattern_), cfa_pattern_), cfa_pattern_);
      break;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }
}

void RawProcessor::ApplyLinearization() {
  auto& pre_debayer_buffer = process_buffer_;
  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    auto& gpu_img = pre_debayer_buffer.GetCUDAImage();
    CUDA::ToLinearRef(gpu_img, raw_processor_, cfa_pattern_);
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = pre_debayer_buffer.GetMetalImage();
    metal::ToLinearRef(gpu_img, raw_processor_, cfa_pattern_);
    return;
#else
    ThrowUnsupportedGPUBackend("ApplyLinearization");
#endif
  }
  auto& cpu_img = pre_debayer_buffer.GetCPUData();
  // Apply "as shot" white balance multipliers for highlight reconstruction
  // This step will stretch the image to original 16-bit range
  // Because 0-65535 is mapped to 0-1 float range, we can also think this as
  // stretching to [0, 1] range
  CPU::ToLinearRef(cpu_img, raw_processor_);
}

void RawProcessor::ApplyDebayer() {
  auto& pre_debayer_buffer = process_buffer_;
  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    auto& gpu_img = pre_debayer_buffer.GetCUDAImage();
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      const int passes = params_.decode_res_ == DecodeRes::FULL ? 3 : 1;
      CUDA::XTransToRGB_Ref(gpu_img, cfa_pattern_.xtrans_pattern, passes);
    } else {
      CUDA::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    }
    if (params_.decode_res_ == DecodeRes::FULL) {
      cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                         raw_data_.sizes.width, raw_data_.sizes.height);
      gpu_img = gpu_img(crop_rect);
    }

    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = pre_debayer_buffer.GetMetalImage();
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      const int passes = params_.decode_res_ == DecodeRes::FULL ? 3 : 1;
      metal::XTransToRGB_Ref(gpu_img, cfa_pattern_.xtrans_pattern, passes);
    } else {
      metal::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern);
    }
    if (params_.decode_res_ == DecodeRes::FULL) {
      cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                         raw_data_.sizes.width, raw_data_.sizes.height);
      metal::MetalImage cropped;
      gpu_img.CropTo(cropped, crop_rect);
      gpu_img = std::move(cropped);
    }
    return;
#else
    ThrowUnsupportedGPUBackend("ApplyDebayer");
#endif
  }
  auto& img = pre_debayer_buffer.GetCPUData();
  if (cfa_pattern_.kind != RawCfaKind::Bayer2x2) {
    throw std::runtime_error("RawProcessor: CPU debayer only supports classic Bayer CFA.");
  }
  CPU::BayerRGGB2RGB_RCD(img);
  // Crop to valid area
  // cv::Rect crop_rect(_raw_data.sizes.raw_inset_crops[0].cleft,
  // _raw_data.sizes.raw_inset_crops[0].ctop,
  //                    _raw_data.sizes.raw_inset_crops[0].cwidth,
  //                    _raw_data.sizes.raw_inset_crops[0].cheight);
  if (params_.decode_res_ == DecodeRes::FULL) {
    cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                       raw_data_.sizes.width, raw_data_.sizes.height);
    img = img(crop_rect);
  }
}

void RawProcessor::ApplyHighlightReconstruct() {
  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    auto& gpu_img = process_buffer_.GetCUDAImage();
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      CUDA::Clamp01(gpu_img);
      return;
    }
    if (!params_.highlights_reconstruct_) {
      CUDA::Clamp01(gpu_img);
      return;
    }
    CUDA::HighlightReconstruct(gpu_img, raw_processor_, cfa_pattern_.bayer_pattern);
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = process_buffer_.GetMetalImage();
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      metal::utils::ClampTexture(gpu_img);
      return;
    }
    if (!params_.highlights_reconstruct_) {
      metal::utils::ClampTexture(gpu_img);
      return;
    }
    metal::HighlightReconstruct(gpu_img, raw_processor_, cfa_pattern_.bayer_pattern);
    return;
#endif
    ThrowUnsupportedGPUBackend("ApplyHighlightReconstruct");
  }

  auto& img = process_buffer_.GetCPUData();
  if (!params_.highlights_reconstruct_) {
    // clamp to [0, 1]
    cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
    return;
  }
  CPU::HighlightReconstruct(img, raw_processor_);
}

void RawProcessor::ApplyGeometricCorrections() {
  // TODO: Add lens distortion correction if needed

  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    auto& gpu_img = process_buffer_.GetCUDAImage();

    switch (raw_data_.sizes.flip) {
      case 3:
        // 180 degree
        CUDA::Rotate180(gpu_img);
        break;
      case 5:
        // Rotate 90 CCW
        CUDA::Rotate90CCW(gpu_img);
        break;
      case 6:
        // Rotate 90 CW
        CUDA::Rotate90CW(gpu_img);
        break;
      default:
        // Do nothing
        break;
    }
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = process_buffer_.GetMetalImage();

    switch (raw_data_.sizes.flip) {
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
    return;
#endif
    ThrowUnsupportedGPUBackend("ApplyGeometricCorrections");
  }

  switch (raw_data_.sizes.flip) {
    case 3:
      // 180 degree
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(), cv::ROTATE_180);
      break;
    case 5:
      // Rotate 90 CCW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 6:
      // Rotate 90 CW
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_CLOCKWISE);
      break;
    default:
      // Do nothing
      break;
  }
}

void RawProcessor::ConvertToWorkingSpace() {
  auto& debayer_buffer = process_buffer_;
  auto  color_coeffs   = raw_data_.color.rgb_cam;
  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    auto& gpu_img = debayer_buffer.GetCUDAImage();
    auto  cam_mul = raw_data_.color.cam_mul;
    CUDA::ApplyInverseCamMul(gpu_img, cam_mul);
    runtime_color_context_.output_in_camera_space_ = true;
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = debayer_buffer.GetMetalImage();
    auto  cam_mul = raw_data_.color.cam_mul;
    metal::ApplyInverseCamMul(gpu_img, cam_mul);
    runtime_color_context_.output_in_camera_space_ = true;
    return;
#endif
    ThrowUnsupportedGPUBackend("ConvertToWorkingSpace");
  }
  auto& img = debayer_buffer.GetCPUData();
  img.convertTo(img, CV_32FC3);
  auto pre_mul   = raw_data_.color.pre_mul;
  auto cam_mul   = raw_data_.color.cam_mul;
  auto wb_coeffs = raw_data_.color.WB_Coeffs;  // EXIF Lightsource Values
  auto cam_xyz   = raw_data_.color.cam_xyz;
  if (!params_.use_camera_wb_) {
    // User specified white balance temperature
    auto user_temp_indices = CPU::GetWBIndicesForTemp(static_cast<float>(params_.user_wb_));
    CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, wb_coeffs, user_temp_indices,
                          params_.user_wb_, cam_xyz);
    runtime_color_context_.output_in_camera_space_ = false;
    return;
  }
  CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, cam_xyz);
  runtime_color_context_.output_in_camera_space_ = false;
}

auto RawProcessor::Process() -> ImageBuffer {
  input_kind_ = ClassifyRawInput(raw_data_);

  // runtime_color_context_ is pre-populated by the caller (via MetadataExtractor).
  // Only update the output color-space flag which depends on the decode pipeline.
  runtime_color_context_.output_in_camera_space_ = false;

  if (input_kind_ == RawInputKind::DebayeredRgb) {
    process_buffer_ = {BuildDirectRgbRgba(raw_data_)};
    runtime_color_context_.output_in_camera_space_ = true;

    if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#if defined(HAVE_CUDA) || defined(HAVE_METAL)
      process_buffer_.SyncToGPU();
      ApplyGeometricCorrections();
      return {std::move(process_buffer_)};
#else
      ThrowUnsupportedGPUBackend("Process");
#endif
    }

    ApplyGeometricCorrections();
    return {std::move(process_buffer_)};
  }

  if (input_kind_ != RawInputKind::BayerRaw) {
    throw std::runtime_error("RawProcessor: " +
                             DescribeUnsupportedRawInput(raw_processor_, input_kind_));
  }

  if (raw_processor_.imgdata.idata.is_foveon != 0U || raw_processor_.imgdata.idata.filters == 0U ||
      raw_processor_.imgdata.idata.filters == 1U || raw_processor_.is_fuji_rotated() != 0) {
    throw std::runtime_error("RawProcessor: " +
                             DescribeUnsupportedRawInput(raw_processor_, input_kind_));
  }

  cfa_pattern_ = ReadLibRawCfaPattern(raw_processor_);
  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 &&
      !IsClassic2x2Bayer(cfa_pattern_.bayer_pattern)) {
    throw std::runtime_error("RawProcessor: unsupported 2x2 CFA pattern " +
                             DescribeBayerPattern(cfa_pattern_.bayer_pattern) + ".");
  }

  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && !IsRGGBPattern(cfa_pattern_.bayer_pattern)) {
    if (params_.gpu_backend_ == RawGpuBackend::CPU) {
      throw std::runtime_error("RawProcessor: CPU backend only supports RGGB Bayer input; got " +
                               DescribeBayerPattern(cfa_pattern_.bayer_pattern) + ".");
    }
  }
  if (cfa_pattern_.kind == RawCfaKind::XTrans6x6 && params_.gpu_backend_ != RawGpuBackend::GPU) {
    throw std::runtime_error("RawProcessor: CPU backend does not support X-Trans CFA input.");
  }

  auto  img_unpacked = raw_data_.raw_image;
  auto& img_sizes    = raw_data_.sizes;

  // LibRaw owns raw_image and frees it during recycle(), so the pipeline must
  // materialize an owned copy before returning ImageBuffer state to callers.
  cv::Mat unpacked_view{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  cv::Mat unpacked_mat = unpacked_view.clone();
  process_buffer_      = {std::move(unpacked_mat)};

  if (params_.gpu_backend_ == RawGpuBackend::GPU) {
#ifdef HAVE_CUDA
    // Keep raw data in 16-bit until it reaches the CUDA linearization stage.
    SetDecodeRes();
    process_buffer_.SyncToGPU();

    ApplyLinearization();
    ApplyHighlightReconstruct();
    ApplyDebayer();
    ConvertToWorkingSpace();

    // Ensure GPU buffer is CV_32FC4 (RGBA float32).
    CUDA::RGBToRGBA(process_buffer_.GetCUDAImage());
    ApplyGeometricCorrections();

    return {std::move(process_buffer_)};
    // process_buffer_.SyncToCPU();
    // process_buffer_.ReleaseGPUData();
#elif defined(HAVE_METAL)
    SetDecodeRes();
    process_buffer_.SyncToGPU();
    ApplyLinearization();
    ApplyHighlightReconstruct();
    ApplyDebayer();
    ConvertToWorkingSpace();
    ApplyGeometricCorrections();
    return {std::move(process_buffer_)};
#else
    ThrowUnsupportedGPUBackend("Process");
#endif
  }

  SetDecodeRes();
  ApplyLinearization();
  ApplyHighlightReconstruct();
  ApplyDebayer();
  ApplyGeometricCorrections();
  ConvertToWorkingSpace();

  cv::Mat final_img = cv::Mat();
  final_img.create(process_buffer_.GetCPUData().rows, process_buffer_.GetCPUData().cols, CV_32FC4);
  cv::cvtColor(process_buffer_.GetCPUData(), final_img, cv::COLOR_RGB2RGBA);
  process_buffer_ = {std::move(final_img)};

  return {std::move(process_buffer_)};
}
}  // namespace puerhlab
