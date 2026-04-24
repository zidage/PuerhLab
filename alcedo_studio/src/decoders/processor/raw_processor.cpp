//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/raw_processor.hpp"

#include <libraw/libraw.h>
#include <libraw/libraw_const.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <stdexcept>
#include <string>

#include "decoders/processor/operators/cpu/color_space_conv.hpp"
#include "decoders/processor/operators/cpu/debayer_rcd.hpp"
#include "decoders/processor/operators/cpu/highlight_reconstruct.hpp"
#include "decoders/processor/operators/cpu/raw_proc_utils.hpp"
#include "decoders/processor/operators/cpu/white_balance.hpp"
#include "decoders/processor/raw_processor_internal.hpp"
#include "image/image_buffer.hpp"

namespace alcedo {
namespace {

auto DecodeResToDownsamplePasses(const DecodeRes decode_res) -> int {
  switch (decode_res) {
    case DecodeRes::FULL:
      return 0;
    case DecodeRes::HALF:
      return 1;
    case DecodeRes::QUARTER:
      return 2;
    case DecodeRes::EIGHTH:
      return 3;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }
}

void NormalizeDecodeResForGpu(const cv::Size& image_size, RawParams& params) {
  const int long_side = std::max(image_size.width, image_size.height);
  if (long_side > 8500 && params.decode_res_ == DecodeRes::QUARTER) {
    params.decode_res_ = DecodeRes::EIGHTH;
  }
}

auto DownsampleRawForGpuInput(const cv::Mat& raw_view, RawCfaPattern& pattern, const int passes)
    -> cv::Mat {
  if (passes <= 0) {
    return raw_view.clone();
  }

  cv::Mat downsampled = DownsampleRaw2x(raw_view, pattern);
  for (int pass = 1; pass < passes; ++pass) {
    downsampled = DownsampleRaw2x(downsampled, pattern);
  }
  return downsampled;
}

[[noreturn]] void ThrowUnsupportedGPUBackend(const char* op_name) {
  throw std::runtime_error(std::string("RawProcessor: ") + op_name +
                           " requires a compiled GPU backend implementation.");
}

auto RawGpuBackendName(const RawGpuBackend backend) -> const char* {
  switch (backend) {
    case RawGpuBackend::CPU:
      return "CPU";
    case RawGpuBackend::GPU:
      return "GPU";
    case RawGpuBackend::CUDA:
      return "CUDA";
    case RawGpuBackend::Metal:
      return "Metal";
    case RawGpuBackend::WebGPU:
      return "WebGPU";
  }
  return "unknown";
}

[[noreturn]] void ThrowUnavailableRawGpuBackend(const RawGpuBackend backend) {
  throw std::runtime_error(std::string("RawProcessor: requested ") + RawGpuBackendName(backend) +
                           " backend is not compiled.");
}

auto CropToActiveArea(const cv::Mat& src, const libraw_image_sizes_t& sizes) -> cv::Mat {
  const cv::Rect active_rect = detail::BuildActiveAreaRect(sizes, src.size());
  return src(active_rect).clone();
}

auto BuildOpaqueRgbaFromRgb(const cv::Mat& rgb) -> cv::Mat {
  cv::Mat rgba;
  cv::cvtColor(rgb, rgba, cv::COLOR_RGB2RGBA);
  return rgba;
}

auto ExtractRgbFromFourChannel(const cv::Mat& src) -> cv::Mat {
  cv::Mat   rgb(src.rows, src.cols, CV_MAKETYPE(src.depth(), 3));
  const int from_to[] = {0, 0, 1, 1, 2, 2};
  cv::mixChannels(&src, 1, &rgb, 1, from_to, 3);
  return rgb;
}

auto BuildDirectRgbRgba(const libraw_rawdata_t& raw_data, const libraw_iparams_t& idata)
    -> cv::Mat {
  const auto& sizes      = raw_data.sizes;
  const int   raw_width  = static_cast<int>(sizes.raw_width);
  const int   raw_height = static_cast<int>(sizes.raw_height);

  if (raw_data.color3_image != nullptr) {
    const size_t row_step = sizes.raw_pitch != 0
                                ? static_cast<size_t>(sizes.raw_pitch)
                                : static_cast<size_t>(raw_width) * sizeof(uint16_t) * 3;
    cv::Mat      view(raw_height, raw_width, CV_16UC3, raw_data.color3_image, row_step);
    cv::Mat      rgb32f;
    CropToActiveArea(view, sizes).convertTo(rgb32f, CV_32FC3, 1.0 / 65535.0);
    return BuildOpaqueRgbaFromRgb(rgb32f);
  }

  if (raw_data.float3_image != nullptr) {
    const size_t row_step = sizes.raw_pitch != 0
                                ? static_cast<size_t>(sizes.raw_pitch)
                                : static_cast<size_t>(raw_width) * sizeof(float) * 3;
    cv::Mat      view(raw_height, raw_width, CV_32FC3, raw_data.float3_image, row_step);
    return BuildOpaqueRgbaFromRgb(CropToActiveArea(view, sizes));
  }

  if (raw_data.color4_image != nullptr && idata.colors == 3) {
    const size_t row_step = sizes.raw_pitch != 0
                                ? static_cast<size_t>(sizes.raw_pitch)
                                : static_cast<size_t>(raw_width) * sizeof(uint16_t) * 4;
    cv::Mat      view(raw_height, raw_width, CV_16UC4, raw_data.color4_image, row_step);
    cv::Mat      rgb16 = ExtractRgbFromFourChannel(CropToActiveArea(view, sizes));
    cv::Mat      rgb32f;
    rgb16.convertTo(rgb32f, CV_32FC3, 1.0 / 65535.0);
    return BuildOpaqueRgbaFromRgb(rgb32f);
  }

  if (raw_data.float4_image != nullptr && idata.colors == 3) {
    const size_t row_step = sizes.raw_pitch != 0
                                ? static_cast<size_t>(sizes.raw_pitch)
                                : static_cast<size_t>(raw_width) * sizeof(float) * 4;
    cv::Mat      view(raw_height, raw_width, CV_32FC4, raw_data.float4_image, row_step);
    return BuildOpaqueRgbaFromRgb(ExtractRgbFromFourChannel(CropToActiveArea(view, sizes)));
  }

  throw std::runtime_error("RawProcessor: direct RGB input is missing a 3-channel source buffer.");
}

auto DescribeUnsupportedRawInput(LibRaw& raw_processor, const RawInputKind input_kind)
    -> std::string {
  const auto& idata   = raw_processor.imgdata.idata;
  const auto& rawdata = raw_processor.imgdata.rawdata;
  if (idata.is_foveon != 0U) {
    return "Foveon/X3F input is not supported by the alcedo raw pipeline.";
  }
  if (raw_processor.is_fuji_rotated() != 0) {
    return "Fuji rotated CFA layouts are not supported.";
  }
  if (rawdata.color4_image != nullptr || rawdata.float4_image != nullptr) {
    if (idata.colors == 3) {
      return "3-color decoded raw input stored in a 4-channel buffer is not supported.";
    }
    return "4-color decoded raw input is not supported.";
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

std::optional<detail::CudaExecutionMode> g_cuda_execution_mode_override;

}  // namespace

namespace detail {

auto SelectCudaExecutionMode(const RawParams& params, const RawCfaPattern& cfa_pattern,
                             const cv::Rect& active_rect) -> CudaExecutionMode {
  if (g_cuda_execution_mode_override.has_value()) {
    return *g_cuda_execution_mode_override;
  }
  if ((params.gpu_backend_ != RawGpuBackend::GPU && params.gpu_backend_ != RawGpuBackend::CUDA) ||
      cfa_pattern.kind != RawCfaKind::Bayer2x2) {
    return CudaExecutionMode::FullFrame;
  }

  const int long_edge = std::max(std::max(active_rect.width, 0), std::max(active_rect.height, 0));
  return long_edge > kCudaTileThresholdLongEdge ? CudaExecutionMode::Tiled
                                                : CudaExecutionMode::FullFrame;
}

void SetCudaExecutionModeOverrideForTesting(const std::optional<CudaExecutionMode>& mode) {
  g_cuda_execution_mode_override = mode;
}

auto GetCudaExecutionModeOverrideForTesting() -> std::optional<CudaExecutionMode> {
  return g_cuda_execution_mode_override;
}

}  // namespace detail

RawProcessor::RawProcessor(const RawParams& params, const libraw_rawdata_t& rawdata,
                           LibRaw& raw_processor, const RawRuntimeColorContext& pre_ctx)
    : params_(params),
      runtime_color_context_(pre_ctx),
      raw_data_(rawdata),
      raw_processor_(raw_processor) {}

void RawProcessor::SetDecodeRes() {
  auto& cpu_data  = process_buffer_.GetCPUData();
  int   long_side = std::max(cpu_data.rows, cpu_data.cols);
  if (long_side > 8500 && params_.decode_res_ == DecodeRes::QUARTER) {
    params_.decode_res_ = DecodeRes::EIGHTH;
  }

  int downsample_passes = 0;
  switch (params_.decode_res_) {
    case DecodeRes::FULL:
      downsample_passes = 0;
      break;
    case DecodeRes::HALF:
      downsample_passes = 1;
      break;
    case DecodeRes::QUARTER:
      downsample_passes = 2;
      break;
    case DecodeRes::EIGHTH:
      downsample_passes = 3;
      break;
    default:
      throw std::runtime_error("RawProcessor: Unknown decode resolution");
  }

  for (int pass = 0; pass < downsample_passes; ++pass) {
    cpu_data = DownsampleRaw2x(cpu_data, cfa_pattern_);
  }
}

void RawProcessor::ApplyLinearization() {
  auto& cpu_img = process_buffer_.GetCPUData();
  CPU::ToLinearRef(cpu_img, raw_processor_);
}

void RawProcessor::ApplyDebayer() {
  auto& img = process_buffer_.GetCPUData();
  if (cfa_pattern_.kind != RawCfaKind::Bayer2x2) {
    throw std::runtime_error("RawProcessor: CPU debayer only supports classic Bayer CFA.");
  }
  CPU::BayerRGGB2RGB_RCD(img);
  const cv::Rect crop_rect =
      detail::BuildDecodeCropRect(raw_data_.sizes, img.size(), params_.decode_res_);
  if (!detail::IsFullImageRect(crop_rect, img.size())) {
    img = img(crop_rect);
  }
}

void RawProcessor::ApplyHighlightReconstruct() {
  auto& img = process_buffer_.GetCPUData();
  if (!params_.highlights_reconstruct_) {
    cv::threshold(img, img, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(img, img, 0.0f, 0.0f, cv::THRESH_TOZERO);
    return;
  }
  CPU::HighlightReconstruct(img, raw_processor_);
}

void RawProcessor::ApplyGeometricCorrections() {
  switch (raw_data_.sizes.flip) {
    case 3:
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(), cv::ROTATE_180);
      break;
    case 5:
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 6:
      cv::rotate(process_buffer_.GetCPUData(), process_buffer_.GetCPUData(),
                 cv::ROTATE_90_CLOCKWISE);
      break;
    default:
      break;
  }
}

void RawProcessor::ConvertToWorkingSpace() {
  auto& img          = process_buffer_.GetCPUData();
  auto  color_coeffs = raw_data_.color.rgb_cam;
  img.convertTo(img, CV_32FC3);
  auto pre_mul   = raw_data_.color.pre_mul;
  auto cam_mul   = raw_data_.color.cam_mul;
  auto wb_coeffs = raw_data_.color.WB_Coeffs;
  auto cam_xyz   = raw_data_.color.cam_xyz;
  if (!params_.use_camera_wb_) {
    auto user_temp_indices = CPU::GetWBIndicesForTemp(static_cast<float>(params_.user_wb_));
    CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, wb_coeffs, user_temp_indices,
                          params_.user_wb_, cam_xyz);
    runtime_color_context_.output_in_camera_space_ = false;
    return;
  }
  CPU::ApplyColorMatrix(img, color_coeffs, pre_mul, cam_mul, cam_xyz);
  runtime_color_context_.output_in_camera_space_ = false;
}

auto RawProcessor::ProcessGpu() -> ImageBuffer {
  switch (params_.gpu_backend_) {
    case RawGpuBackend::CPU:
      ThrowUnsupportedGPUBackend("Process");
    case RawGpuBackend::CUDA:
#ifdef HAVE_CUDA
      return ProcessCuda();
#else
      ThrowUnavailableRawGpuBackend(params_.gpu_backend_);
#endif
    case RawGpuBackend::Metal:
#ifdef HAVE_METAL
      return ProcessMetal();
#else
      ThrowUnavailableRawGpuBackend(params_.gpu_backend_);
#endif
    case RawGpuBackend::WebGPU:
#ifdef HAVE_WEBGPU
      return ProcessWebGpu();
#else
      ThrowUnavailableRawGpuBackend(params_.gpu_backend_);
#endif
    case RawGpuBackend::GPU:
#ifdef HAVE_CUDA
      return ProcessCuda();
#elif defined(HAVE_METAL)
      return ProcessMetal();
#elif defined(HAVE_WEBGPU)
      return ProcessWebGpu();
#else
      ThrowUnsupportedGPUBackend("Process");
#endif
  }
  ThrowUnsupportedGPUBackend("Process");
}

auto RawProcessor::Process() -> ImageBuffer {
  input_kind_ = ClassifyRawInput(raw_data_, raw_processor_.imgdata.idata);
  runtime_color_context_.output_in_camera_space_ = false;
  gpu_input_downsample_passes_                   = 0;

  if (input_kind_ == RawInputKind::DebayeredRgb) {
    params_.highlights_reconstruct_ = false;
    process_buffer_                 = {BuildDirectRgbRgba(raw_data_, raw_processor_.imgdata.idata)};
    runtime_color_context_.output_in_camera_space_ = true;

    if (IsRawGpuBackend(params_.gpu_backend_)) {
      return ProcessGpu();
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
  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && !IsClassic2x2Bayer(cfa_pattern_.bayer_pattern)) {
    throw std::runtime_error("RawProcessor: unsupported 2x2 CFA pattern " +
                             DescribeBayerPattern(cfa_pattern_.bayer_pattern) + ".");
  }

  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && !IsRGGBPattern(cfa_pattern_.bayer_pattern) &&
      params_.gpu_backend_ == RawGpuBackend::CPU) {
    throw std::runtime_error("RawProcessor: CPU backend only supports RGGB Bayer input; got " +
                             DescribeBayerPattern(cfa_pattern_.bayer_pattern) + ".");
  }
  if (cfa_pattern_.kind == RawCfaKind::XTrans6x6 && !IsRawGpuBackend(params_.gpu_backend_)) {
    throw std::runtime_error("RawProcessor: CPU backend does not support X-Trans CFA input.");
  }

  if (IsRawGpuBackend(params_.gpu_backend_)) {
    NormalizeDecodeResForGpu(cv::Size(static_cast<int>(raw_data_.sizes.raw_width),
                                      static_cast<int>(raw_data_.sizes.raw_height)),
                             params_);
    gpu_input_downsample_passes_ = DecodeResToDownsamplePasses(params_.decode_res_);

    cv::Mat unpacked_view{raw_data_.sizes.raw_height, raw_data_.sizes.raw_width, CV_16UC1,
                          raw_data_.raw_image};
    process_buffer_ = {
        DownsampleRawForGpuInput(unpacked_view, cfa_pattern_, gpu_input_downsample_passes_)};
    return ProcessGpu();
  }

  cv::Mat unpacked_view{raw_data_.sizes.raw_height, raw_data_.sizes.raw_width, CV_16UC1,
                        raw_data_.raw_image};
  process_buffer_ = {unpacked_view.clone()};

  SetDecodeRes();
  ApplyLinearization();
  ApplyHighlightReconstruct();
  ApplyDebayer();
  ApplyGeometricCorrections();
  ConvertToWorkingSpace();

  cv::Mat final_img;
  final_img.create(process_buffer_.GetCPUData().rows, process_buffer_.GetCPUData().cols, CV_32FC4);
  cv::cvtColor(process_buffer_.GetCPUData(), final_img, cv::COLOR_RGB2RGBA);
  process_buffer_ = {std::move(final_img)};

  return {std::move(process_buffer_)};
}

}  // namespace alcedo
