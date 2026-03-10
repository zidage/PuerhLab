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
#endif
#ifdef HAVE_METAL
#include "decoders/processor/operators/gpu/metal_cvt_ref_space.hpp"
#include "decoders/processor/operators/gpu/metal_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/metal_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/metal_to_linear_ref.hpp"
#include "metal/metal_utils/metal_convert_utils.hpp"
#endif
#include "image/image_buffer.hpp"

namespace puerhlab {
namespace {
template <typename T>
auto DownsampleBayerRGGB2xTyped(const cv::Mat& src) -> cv::Mat {
  const int out_rows = src.rows / 2;
  const int out_cols = src.cols / 2;
  cv::Mat   dst(out_rows, out_cols, src.type());

  for (int y = 0; y < out_rows; ++y) {
    const T* row0 = src.ptr<T>(2 * y);
    const T* row1 = src.ptr<T>(2 * y + 1);
    T*       drow = dst.ptr<T>(y);
    for (int x = 0; x < out_cols; ++x) {
      const int sx = 2 * x;
      if ((y & 1) == 0) {
        // R G
        // G B
        drow[x] = (x & 1) == 0 ? row0[sx] : row0[sx + 1];
      } else {
        drow[x] = (x & 1) == 0 ? row1[sx] : row1[sx + 1];
      }
    }
  }

  return dst;
}

auto DownsampleBayerRGGB2x(const cv::Mat& src) -> cv::Mat {
  switch (src.type()) {
    case CV_32FC1:
      return DownsampleBayerRGGB2xTyped<float>(src);
    case CV_16UC1:
      return DownsampleBayerRGGB2xTyped<uint16_t>(src);
    default:
      throw std::runtime_error("RawProcessor: Unsupported Bayer type for downsample");
  }
}

void ThrowUnsupportedGPUBackend(const char* op_name) {
#ifdef HAVE_METAL
  throw std::runtime_error(std::string("RawProcessor: ") + op_name +
                           " is not implemented for the Metal GPU backend.");
#else
  throw std::runtime_error(std::string("RawProcessor: ") + op_name +
                           " requires a compiled GPU backend implementation.");
#endif
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
      // No changes needed for full resolution
      break;
    case DecodeRes::HALF:
      // Downscale by factor of 2
      cpu_data = DownsampleBayerRGGB2x(cpu_data);
      break;
    case DecodeRes::QUARTER:
      // Downscale by factor of 4
      cpu_data = DownsampleBayerRGGB2x(DownsampleBayerRGGB2x(cpu_data));
      break;
    case DecodeRes::EIGHTH:
      // Downscale by factor of 8
      cpu_data = DownsampleBayerRGGB2x(DownsampleBayerRGGB2x(DownsampleBayerRGGB2x(cpu_data)));
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
    CUDA::ToLinearRef(gpu_img, raw_processor_);
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = pre_debayer_buffer.GetMetalImage();
    metal::ToLinearRef(gpu_img, raw_processor_);
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
    CUDA::BayerRGGB2RGB_RCD(gpu_img);
    if (params_.decode_res_ == DecodeRes::FULL) {
      cv::Rect crop_rect(raw_data_.sizes.left_margin, raw_data_.sizes.top_margin,
                         raw_data_.sizes.width, raw_data_.sizes.height);
      gpu_img = gpu_img(crop_rect);
    }

    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = pre_debayer_buffer.GetMetalImage();
    metal::BayerRGGB2RGB_RCD(gpu_img);
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
    if (!params_.highlights_reconstruct_) {
      CUDA::Clamp01(gpu_img);
      return;
    }
    CUDA::HighlightReconstruct(gpu_img, raw_processor_);
    return;
#elif defined(HAVE_METAL)
    auto& gpu_img = process_buffer_.GetMetalImage();
    if (!params_.highlights_reconstruct_) {
      metal::utils::ClampTexture(gpu_img);
      return;
    }
    metal::HighlightReconstruct(gpu_img, raw_processor_);
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
  auto    img_unpacked = raw_data_.raw_image;
  auto&   img_sizes    = raw_data_.sizes;

  cv::Mat unpacked_mat{img_sizes.raw_height, img_sizes.raw_width, CV_16UC1, img_unpacked};
  process_buffer_                                = {std::move(unpacked_mat)};

  // runtime_color_context_ is pre-populated by the caller (via MetadataExtractor).
  // Only update the output color-space flag which depends on the decode pipeline.
  runtime_color_context_.output_in_camera_space_ = false;

  // std::cout << _raw_processor.COLOR(0, 0) << " " << _raw_processor.COLOR(0, 1) << " "
  //           << _raw_processor.COLOR(1, 0) << " " << _raw_processor.COLOR(1, 1) << "\n";
  CV_Assert(raw_processor_.COLOR(0, 0) == 0 && raw_processor_.COLOR(0, 1) == 1 &&
            raw_processor_.COLOR(1, 0) == 3 && raw_processor_.COLOR(1, 1) == 2);

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

  // CPU pipeline expects float32 input in [0, 1].
  process_buffer_.GetCPUData().convertTo(process_buffer_.GetCPUData(), CV_32FC1, 1.0f / 65535.0f);

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
