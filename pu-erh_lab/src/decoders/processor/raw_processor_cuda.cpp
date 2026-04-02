//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/raw_processor.hpp"

#ifdef HAVE_CUDA

#include <chrono>
#include <iostream>
#include <vector>

#include <opencv2/core/cuda.hpp>

#include "decoders/processor/operators/gpu/cuda_color_space_conv.hpp"
#include "decoders/processor/operators/gpu/cuda_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"
#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"
#include "decoders/processor/operators/gpu/cuda_rotate.hpp"
#include "decoders/processor/operators/gpu/cuda_white_balance.hpp"
#include "decoders/processor/operators/gpu/cuda_xtrans_interpolate.hpp"
#include "decoders/processor/raw_processor_internal.hpp"

namespace puerhlab {
namespace {

struct CudaTileJob {
  cv::Rect source_rect;
  cv::Rect inner_rect_in_tile;
  cv::Rect output_rect;
};

auto ShiftBayerPattern(const BayerPattern2x2& pattern, const int y_offset, const int x_offset)
    -> BayerPattern2x2 {
  BayerPattern2x2 shifted = {};
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const int idx      = BayerCellIndex(y, x);
      shifted.raw_fc[idx] = RawColorAt(pattern, y + y_offset, x + x_offset);
      shifted.rgb_fc[idx] = RgbColorAt(pattern, y + y_offset, x + x_offset);
    }
  }
  return shifted;
}

auto BuildTileJobs(const cv::Rect& active_rect, const cv::Size& full_size) -> std::vector<CudaTileJob> {
  std::vector<CudaTileJob> jobs;
  for (int y = 0; y < active_rect.height; y += detail::kCudaTileInnerSize) {
    const int inner_h = std::min(detail::kCudaTileInnerSize, active_rect.height - y);
    for (int x = 0; x < active_rect.width; x += detail::kCudaTileInnerSize) {
      const int inner_w = std::min(detail::kCudaTileInnerSize, active_rect.width - x);

      const cv::Rect inner_abs(active_rect.x + x, active_rect.y + y, inner_w, inner_h);
      const int      src_x = std::max(0, inner_abs.x - detail::kCudaTileHaloSize);
      const int      src_y = std::max(0, inner_abs.y - detail::kCudaTileHaloSize);
      const int src_r = std::min(full_size.width, inner_abs.x + inner_abs.width + detail::kCudaTileHaloSize);
      const int src_b =
          std::min(full_size.height, inner_abs.y + inner_abs.height + detail::kCudaTileHaloSize);

      const cv::Rect source_rect(src_x, src_y, src_r - src_x, src_b - src_y);
      jobs.push_back({
          .source_rect       = source_rect,
          .inner_rect_in_tile = cv::Rect(inner_abs.x - source_rect.x, inner_abs.y - source_rect.y,
                                         inner_abs.width, inner_abs.height),
          .output_rect       = cv::Rect(x, y, inner_abs.width, inner_abs.height),
      });
    }
  }
  return jobs;
}

void ApplyCudaGeometricCorrections(cv::cuda::GpuMat& gpu_img, const int flip,
                                   cv::cuda::Stream* stream) {
  switch (flip) {
    case 3:
      CUDA::Rotate180(gpu_img, stream);
      break;
    case 5:
      CUDA::Rotate90CCW(gpu_img, stream);
      break;
    case 6:
      CUDA::Rotate90CW(gpu_img, stream);
      break;
    default:
      break;
  }
}

}  // namespace

auto RawProcessor::ProcessCudaFullFrame() -> ImageBuffer {
  process_buffer_.SyncToGPU();
  process_buffer_.ReleaseCPUData();

  cv::cuda::Stream        stream;
  CUDA::RcdWorkspace       rcd_workspace;
  CUDA::HighlightWorkspace highlight_workspace;
  auto&                    gpu_img   = process_buffer_.GetCUDAImage();
  const cv::Rect crop_rect =
      detail::BuildDecodeCropRect(raw_data_.sizes, gpu_img.size(), params_.decode_res_);

  CUDA::ToLinearRef(gpu_img, raw_processor_, cfa_pattern_, &stream);

  if (cfa_pattern_.kind == RawCfaKind::Bayer2x2 && params_.highlights_reconstruct_) {
    CUDA::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern, &rcd_workspace, &stream);
    if (!detail::IsFullImageRect(crop_rect, gpu_img.size())) {
      gpu_img = gpu_img(crop_rect);
    }
    CUDA::HighlightReconstruct(gpu_img, raw_processor_, &highlight_workspace, &stream);
  } else {
    CUDA::Clamp01(gpu_img, &stream);
    if (cfa_pattern_.kind == RawCfaKind::XTrans6x6) {
      const int passes = params_.decode_res_ == DecodeRes::FULL ? 3 : 1;
      stream.waitForCompletion();
      CUDA::XTransToRGB_Ref(gpu_img, cfa_pattern_.xtrans_pattern, passes);
    } else {
      CUDA::Bayer2x2ToRGB_RCD(gpu_img, cfa_pattern_.bayer_pattern, &rcd_workspace, &stream);
    }
    if (!detail::IsFullImageRect(crop_rect, gpu_img.size())) {
      gpu_img = gpu_img(crop_rect);
    }
  }

  CUDA::ApplyInverseCamMul(gpu_img, raw_data_.color.cam_mul, &stream);
  runtime_color_context_.output_in_camera_space_ = true;
  CUDA::RGBToRGBA(gpu_img, &stream);
  ApplyCudaGeometricCorrections(gpu_img, raw_data_.sizes.flip, &stream);
  stream.waitForCompletion();

  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessCudaTiled() -> ImageBuffer {
  process_buffer_.SyncToGPU();
  process_buffer_.ReleaseCPUData();

  cv::cuda::Stream         stream;
  CUDA::RcdWorkspace       rcd_workspace;
  CUDA::HighlightWorkspace highlight_workspace;
  auto&                    linear_raw = process_buffer_.GetCUDAImage();

  CUDA::ToLinearRef(linear_raw, raw_processor_, cfa_pattern_, &stream);

  const cv::Rect active_rect =
      detail::BuildDecodeCropRect(raw_data_.sizes, linear_raw.size(), params_.decode_res_);
  auto jobs = BuildTileJobs(active_rect, linear_raw.size());

  CUDA::HighlightCorrection   correction = CUDA::BuildHighlightCorrection(raw_processor_);
  CUDA::HighlightAccumulation accumulation;

  if (params_.highlights_reconstruct_) {
    for (const auto& job : jobs) {
      cv::cuda::GpuMat tile_raw;
      linear_raw(job.source_rect).copyTo(tile_raw, stream);
      const BayerPattern2x2 tile_pattern =
          ShiftBayerPattern(cfa_pattern_.bayer_pattern, job.source_rect.y, job.source_rect.x);
      CUDA::Bayer2x2ToRGB_RCD(tile_raw, tile_pattern, &rcd_workspace, &stream);
      CUDA::AccumulateHighlightStats(tile_raw, correction, job.inner_rect_in_tile, highlight_workspace,
                                     accumulation, &stream);
    }
    CUDA::FinalizeHighlightCorrection(accumulation, correction);
  }

  cv::cuda::GpuMat output(active_rect.height, active_rect.width, CV_32FC4);
  for (const auto& job : jobs) {
    cv::cuda::GpuMat tile_raw;
    linear_raw(job.source_rect).copyTo(tile_raw, stream);
    const BayerPattern2x2 tile_pattern =
        ShiftBayerPattern(cfa_pattern_.bayer_pattern, job.source_rect.y, job.source_rect.x);
    if (params_.highlights_reconstruct_) {
      CUDA::Bayer2x2ToRGB_RCD(tile_raw, tile_pattern, &rcd_workspace, &stream);
      CUDA::ApplyHighlightCorrection(tile_raw, correction, &highlight_workspace, &stream);
    } else {
      CUDA::Clamp01(tile_raw, &stream);
      CUDA::Bayer2x2ToRGB_RCD(tile_raw, tile_pattern, &rcd_workspace, &stream);
    }
    CUDA::ApplyInverseCamMul(tile_raw, raw_data_.color.cam_mul, &stream);
    CUDA::RGBToRGBA(tile_raw, &stream);
    tile_raw(job.inner_rect_in_tile).copyTo(output(job.output_rect), stream);
  }

  runtime_color_context_.output_in_camera_space_ = true;
  ApplyCudaGeometricCorrections(output, raw_data_.sizes.flip, &stream);
  stream.waitForCompletion();
  process_buffer_ = {std::move(output)};
  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessDirectRgbCuda() -> ImageBuffer {
  process_buffer_.SyncToGPU();
  process_buffer_.ReleaseCPUData();

  cv::cuda::Stream stream;
  auto&            gpu_img = process_buffer_.GetCUDAImage();
  ApplyCudaGeometricCorrections(gpu_img, raw_data_.sizes.flip, &stream);
  stream.waitForCompletion();
  return {std::move(process_buffer_)};
}

auto RawProcessor::ProcessCuda() -> ImageBuffer {
  using clock = std::chrono::high_resolution_clock;
  const auto start = clock::now();

  if (input_kind_ == RawInputKind::DebayeredRgb) {
    auto out = ProcessDirectRgbCuda();
    const auto end = clock::now();
    std::cout << "[LOG] RAW decoding takes: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    return out;
  }

  SetDecodeRes();
  const cv::Rect active_rect =
      detail::BuildDecodeCropRect(raw_data_.sizes,
                                  cv::Size(process_buffer_.GetCPUData().cols, process_buffer_.GetCPUData().rows),
                                  params_.decode_res_);
  const detail::CudaExecutionMode mode =
      detail::SelectCudaExecutionMode(params_, cfa_pattern_, active_rect);

  ImageBuffer out = mode == detail::CudaExecutionMode::Tiled ? ProcessCudaTiled()
                                                             : ProcessCudaFullFrame();
  const auto end = clock::now();
  std::cout << "[LOG] RAW decoding takes: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
  return out;
}

}  // namespace puerhlab

#endif
