//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace alcedo {
namespace CUDA {
namespace {

constexpr float kHilightMagic   = 0.987f;
constexpr int   kDilateRadius   = 3;
constexpr int   kNeutralRadius  = 2;
constexpr int   kMaskPlanes     = 8;
constexpr int   kPlaneBaseR     = 0;
constexpr int   kPlaneBaseG     = 1;
constexpr int   kPlaneBaseB     = 2;
constexpr int   kPlaneDilatedR  = 3;
constexpr int   kPlaneDilatedG  = 4;
constexpr int   kPlaneDilatedB  = 5;
constexpr int   kPlaneBaseMulti = 6;
constexpr int   kPlaneDilMulti  = 7;

struct HighlightCorrectionParams {
  float clips[4];
  float clipdark[4];
  float chrominance[4];
};

auto GetCudaStream(cv::cuda::Stream* stream) -> cudaStream_t {
  if (stream == nullptr) {
    return nullptr;
  }
  return cv::cuda::StreamAccessor::getStream(*stream);
}

void WaitForStream(cv::cuda::Stream* stream) {
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    stream->waitForCompletion();
  }
}

auto NormalizeInnerRegion(const cv::Rect& inner_region, const cv::Size& size) -> cv::Rect {
  if (inner_region.width <= 0 || inner_region.height <= 0) {
    return {0, 0, size.width, size.height};
  }
  return inner_region & cv::Rect(0, 0, size.width, size.height);
}

auto ToParams(const HighlightCorrection& correction) -> HighlightCorrectionParams {
  HighlightCorrectionParams params = {};
  for (int i = 0; i < 4; ++i) {
    params.clips[i]       = correction.clips[i];
    params.clipdark[i]    = correction.clipdark[i];
    params.chrominance[i] = correction.chrominance[i];
  }
  return params;
}

auto BuildInverseCamMulScale(const float* cam_mul) -> float3 {
  const float green = std::max(cam_mul[1], 1e-6f);
  return make_float3(green / std::max(cam_mul[0], 1e-6f), 1.0f,
                     green / std::max(cam_mul[2], 1e-6f));
}

__device__ __forceinline__ float Cube(const float value) { return value * value * value; }

__device__ __forceinline__ float3 ClampRgb(const float3& value) {
  return make_float3(fminf(1.0f, fmaxf(0.0f, value.x)), fminf(1.0f, fmaxf(0.0f, value.y)),
                     fminf(1.0f, fmaxf(0.0f, value.z)));
}

__device__ __forceinline__ float3 MaxRgb(const float3& value) {
  return make_float3(fmaxf(0.0f, value.x), fmaxf(0.0f, value.y), fmaxf(0.0f, value.z));
}

__device__ __forceinline__ float3 CalcRefavg(const cv::cuda::PtrStepSz<float3> input, const int row,
                                             const int col) {
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dymin = max(0, row - 1);
  const int dxmin = max(0, col - 1);
  const int dymax = min(input.rows - 1, row + 1);
  const int dxmax = min(input.cols - 1, col + 1);

  for (int dy = dymin; dy <= dymax; ++dy) {
    const float3* row_ptr = input.ptr(dy);
    for (int dx = dxmin; dx <= dxmax; ++dx) {
      const float3 sample = MaxRgb(row_ptr[dx]);
      mean[0] += sample.x;
      mean[1] += sample.y;
      mean[2] += sample.z;
      cnt[0] += 1.0f;
      cnt[1] += 1.0f;
      cnt[2] += 1.0f;
    }
  }

  for (int c = 0; c < 3; ++c) {
    mean[c] = (cnt[c] > 0.0f) ? cbrtf(mean[c] / cnt[c]) : 0.0f;
  }

  return make_float3(Cube(0.5f * (mean[1] + mean[2])), Cube(0.5f * (mean[0] + mean[2])),
                     Cube(0.5f * (mean[0] + mean[1])));
}

__device__ __forceinline__ uint8_t DilateMaskAt(const uint8_t* plane, const int width,
                                                const int height, const int row, const int col,
                                                const int radius) {
  const int y0 = max(0, row - radius);
  const int x0 = max(0, col - radius);
  const int y1 = min(height - 1, row + radius);
  const int x1 = min(width - 1, col + radius);

  for (int y = y0; y <= y1; ++y) {
    const int row_offset = y * width;
    for (int x = x0; x <= x1; ++x) {
      if (plane[row_offset + x] != 0) {
        return 1;
      }
    }
  }

  return 0;
}

__device__ __forceinline__ int CountClippedChannels(const float3& pixel, const float* clips) {
  int count = 0;
  count += pixel.x >= clips[0] ? 1 : 0;
  count += pixel.y >= clips[1] ? 1 : 0;
  count += pixel.z >= clips[2] ? 1 : 0;
  return count;
}

__device__ __forceinline__ float LocalNeutralTarget(const cv::cuda::PtrStepSz<float3> input,
                                                    const uint8_t* multi_mask, const int row,
                                                    const int col) {
  const int y0 = max(0, row - kNeutralRadius);
  const int x0 = max(0, col - kNeutralRadius);
  const int y1 = min(input.rows - 1, row + kNeutralRadius);
  const int x1 = min(input.cols - 1, col + kNeutralRadius);

  float neutral_sum = 0.0f;
  float neutral_cnt = 0.0f;
  for (int y = y0; y <= y1; ++y) {
    const float3* row_ptr = input.ptr(y);
    const int     row_off = y * input.cols;
    for (int x = x0; x <= x1; ++x) {
      if (multi_mask[row_off + x] == 0) {
        continue;
      }
      const float3 sample = MaxRgb(row_ptr[x]);
      neutral_sum += fmaxf(sample.x, fmaxf(sample.y, sample.z));
      neutral_cnt += 1.0f;
    }
  }

  if (neutral_cnt > 0.0f) {
    return neutral_sum / neutral_cnt;
  }

  const float3 current = MaxRgb(input.ptr(row)[col]);
  return fmaxf(current.x, fmaxf(current.y, current.z));
}

__global__ void Clamp01KernelGray(cv::cuda::PtrStep<float> img, const int width, const int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height) {
    return;
  }

  img(row, col) = fminf(1.0f, fmaxf(0.0f, img(row, col)));
}

__global__ void Clamp01KernelRgb(cv::cuda::PtrStepSz<float3> img) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= img.cols || row >= img.rows) {
    return;
  }

  img.ptr(row)[col] = ClampRgb(img.ptr(row)[col]);
}

__global__ void BuildMaskKernel(const cv::cuda::PtrStepSz<float3> input, uint8_t* mask_buf,
                                int* anyclipped, HighlightCorrectionParams params) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }

  const int    idx   = row * input.cols + col;
  const float3 pixel = MaxRgb(input.ptr(row)[col]);
  const int    count = CountClippedChannels(pixel, params.clips);
  const int    size  = input.rows * input.cols;

  mask_buf[kPlaneBaseR * size + idx]     = pixel.x >= params.clips[0] ? 1 : 0;
  mask_buf[kPlaneBaseG * size + idx]     = pixel.y >= params.clips[1] ? 1 : 0;
  mask_buf[kPlaneBaseB * size + idx]     = pixel.z >= params.clips[2] ? 1 : 0;
  mask_buf[kPlaneBaseMulti * size + idx] = count >= 2 ? 1 : 0;

  if (count > 0) {
    atomicExch(anyclipped, 1);
  }
}

__global__ void DilateMaskKernel(const uint8_t* mask_buf, uint8_t* dilated_mask_buf, const int width,
                                 const int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height) {
    return;
  }

  const int size = width * height;
  const int idx  = row * width + col;
  dilated_mask_buf[kPlaneDilatedR * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseR * size, width, height, row, col, kDilateRadius);
  dilated_mask_buf[kPlaneDilatedG * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseG * size, width, height, row, col, kDilateRadius);
  dilated_mask_buf[kPlaneDilatedB * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseB * size, width, height, row, col, kDilateRadius);
  dilated_mask_buf[kPlaneDilMulti * size + idx] =
      DilateMaskAt(mask_buf + kPlaneBaseMulti * size, width, height, row, col, kDilateRadius);
}

__global__ void ChrominanceAccumulateKernel(const cv::cuda::PtrStepSz<float3> input,
                                            const uint8_t* dilated_mask_buf, float* sums,
                                            float* cnts, HighlightCorrectionParams params,
                                            const int x0, const int y0, const int x1,
                                            const int y1) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }
  if (col < x0 || col >= x1 || row < y0 || row >= y1) {
    return;
  }

  const int    size  = input.rows * input.cols;
  const int    idx   = row * input.cols + col;
  const float3 pixel = MaxRgb(input.ptr(row)[col]);
  const float3 ref   = CalcRefavg(input, row, col);

  if (dilated_mask_buf[kPlaneDilatedR * size + idx] && pixel.x > params.clipdark[0] &&
      pixel.x < params.clips[0]) {
    atomicAdd(&sums[0], pixel.x - ref.x);
    atomicAdd(&cnts[0], 1.0f);
  }
  if (dilated_mask_buf[kPlaneDilatedG * size + idx] && pixel.y > params.clipdark[1] &&
      pixel.y < params.clips[1]) {
    atomicAdd(&sums[1], pixel.y - ref.y);
    atomicAdd(&cnts[1], 1.0f);
  }
  if (dilated_mask_buf[kPlaneDilatedB * size + idx] && pixel.z > params.clipdark[2] &&
      pixel.z < params.clips[2]) {
    atomicAdd(&sums[2], pixel.z - ref.z);
    atomicAdd(&cnts[2], 1.0f);
  }
}

__global__ void HighlightReconstructKernel(const cv::cuda::PtrStepSz<float3> input,
                                           cv::cuda::PtrStepSz<float3> output,
                                           const uint8_t* dilated_mask_buf,
                                           HighlightCorrectionParams params) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }

  const float3 pixel = MaxRgb(input.ptr(row)[col]);
  const int    count = CountClippedChannels(pixel, params.clips);

  if (count == 0) {
    output.ptr(row)[col] = pixel;
    return;
  }

  float3       result = pixel;
  const float3 ref    = CalcRefavg(input, row, col);

  if (pixel.x >= params.clips[0]) {
    result.x = fmaxf(pixel.x, ref.x + params.chrominance[0]);
  }
  if (pixel.y >= params.clips[1]) {
    result.y = fmaxf(pixel.y, ref.y + params.chrominance[1]);
  }
  if (pixel.z >= params.clips[2]) {
    result.z = fmaxf(pixel.z, ref.z + params.chrominance[2]);
  }


  output.ptr(row)[col] = result;
}

__global__ void ClampAndPackRGBAKernel(const cv::cuda::PtrStepSz<float3> input,
                                       cv::cuda::PtrStepSz<float4> output, const float3 gain) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }

  const float3 pixel = ClampRgb(input.ptr(row)[col]);
  output.ptr(row)[col] =
      make_float4(pixel.x * gain.x, pixel.y * gain.y, pixel.z * gain.z, 1.0f);
}

__global__ void HighlightReconstructAndPackRGBAKernel(const cv::cuda::PtrStepSz<float3> input,
                                                      cv::cuda::PtrStepSz<float4> output,
                                                      const uint8_t* dilated_mask_buf,
                                                      HighlightCorrectionParams params,
                                                      const float3 gain) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }

  const float3 pixel = MaxRgb(input.ptr(row)[col]);
  const int    count = CountClippedChannels(pixel, params.clips);

  float3 result = pixel;
  if (count != 0) {
    const float3 ref = CalcRefavg(input, row, col);
    if (pixel.x >= params.clips[0]) {
      result.x = fmaxf(pixel.x, ref.x + params.chrominance[0]);
    }
    if (pixel.y >= params.clips[1]) {
      result.y = fmaxf(pixel.y, ref.y + params.chrominance[1]);
    }
    if (pixel.z >= params.clips[2]) {
      result.z = fmaxf(pixel.z, ref.z + params.chrominance[2]);
    }
  }

  output.ptr(row)[col] =
      make_float4(result.x * gain.x, result.y * gain.y, result.z * gain.z, 1.0f);
}

}  // namespace

HighlightWorkspace::HighlightWorkspace() = default;

HighlightWorkspace::~HighlightWorkspace() { Release(); }

HighlightWorkspace::HighlightWorkspace(HighlightWorkspace&& other) noexcept
    : mask_buf_(std::exchange(other.mask_buf_, nullptr)),
      dilated_mask_(std::exchange(other.dilated_mask_, nullptr)),
      anyclipped_(std::exchange(other.anyclipped_, nullptr)),
      sums_(std::exchange(other.sums_, nullptr)),
      cnts_(std::exchange(other.cnts_, nullptr)),
      mask_capacity_(std::exchange(other.mask_capacity_, 0)),
      result_(std::move(other.result_)) {}

auto HighlightWorkspace::operator=(HighlightWorkspace&& other) noexcept -> HighlightWorkspace& {
  if (this != &other) {
    Release();
    mask_buf_      = std::exchange(other.mask_buf_, nullptr);
    dilated_mask_  = std::exchange(other.dilated_mask_, nullptr);
    anyclipped_    = std::exchange(other.anyclipped_, nullptr);
    sums_          = std::exchange(other.sums_, nullptr);
    cnts_          = std::exchange(other.cnts_, nullptr);
    mask_capacity_ = std::exchange(other.mask_capacity_, 0);
    result_        = std::move(other.result_);
  }
  return *this;
}

void HighlightWorkspace::Reserve(int width, int height) {
  const size_t pixels = static_cast<size_t>(std::max(width, 0)) * static_cast<size_t>(std::max(height, 0));
  if (pixels == 0) {
    result_.release();
    return;
  }

  if (mask_capacity_ < pixels) {
    Release();
    CUDA_CHECK(cudaMalloc(&mask_buf_, static_cast<size_t>(kMaskPlanes) * pixels * sizeof(uint8_t)));
    CUDA_CHECK(
        cudaMalloc(&dilated_mask_, static_cast<size_t>(kMaskPlanes) * pixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&anyclipped_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sums_, sizeof(float) * 4));
    CUDA_CHECK(cudaMalloc(&cnts_, sizeof(float) * 4));
    mask_capacity_ = pixels;
  }

  result_.create(height, width, CV_32FC3);
}

void HighlightWorkspace::Release() {
  if (cnts_ != nullptr) {
    CUDA_CHECK(cudaFree(cnts_));
    cnts_ = nullptr;
  }
  if (sums_ != nullptr) {
    CUDA_CHECK(cudaFree(sums_));
    sums_ = nullptr;
  }
  if (anyclipped_ != nullptr) {
    CUDA_CHECK(cudaFree(anyclipped_));
    anyclipped_ = nullptr;
  }
  if (dilated_mask_ != nullptr) {
    CUDA_CHECK(cudaFree(dilated_mask_));
    dilated_mask_ = nullptr;
  }
  if (mask_buf_ != nullptr) {
    CUDA_CHECK(cudaFree(mask_buf_));
    mask_buf_ = nullptr;
  }
  mask_capacity_ = 0;
  result_.release();
}

auto BuildHighlightCorrection(LibRaw& raw_processor) -> HighlightCorrection {
  const float* cam_mul = raw_processor.imgdata.color.cam_mul;
  const float  green   = std::max(cam_mul[1], 1e-6f);

  HighlightCorrection correction;
  correction.clips[0]    = kHilightMagic * (cam_mul[0] / green);
  correction.clips[1]    = kHilightMagic;
  correction.clips[2]    = kHilightMagic * (cam_mul[2] / green);
  correction.clipdark[0] = 0.03f * correction.clips[0];
  correction.clipdark[1] = 0.125f * correction.clips[1];
  correction.clipdark[2] = 0.03f * correction.clips[2];
  return correction;
}

void FinalizeHighlightCorrection(const HighlightAccumulation& accumulation,
                                 HighlightCorrection& correction) {
  correction.any_clipped = accumulation.any_clipped;
  if (!accumulation.any_clipped) {
    correction.chrominance = {};
    return;
  }

  for (int c = 0; c < 3; ++c) {
    correction.chrominance[c] =
        accumulation.cnts[c] > 0.0 ? static_cast<float>(accumulation.sums[c] / accumulation.cnts[c])
                                    : 0.0f;
  }
}

void Clamp01(cv::cuda::GpuMat& img, cv::cuda::Stream* stream) {
  if (img.type() != CV_32FC1 && img.type() != CV_32FC3) {
    throw std::runtime_error("CUDA::Clamp01: only CV_32FC1/CV_32FC3 are supported");
  }

  const dim3 threads(32, 8);
  const dim3 blocks((img.cols + threads.x - 1) / threads.x, (img.rows + threads.y - 1) / threads.y);
  const auto cuda_stream = GetCudaStream(stream);

  if (img.type() == CV_32FC1) {
    Clamp01KernelGray<<<blocks, threads, 0, cuda_stream>>>(img, img.cols, img.rows);
  } else {
    Clamp01KernelRgb<<<blocks, threads, 0, cuda_stream>>>(img);
  }

  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    WaitForStream(stream);
  }
}

void AccumulateHighlightStats(const cv::cuda::GpuMat& img, const HighlightCorrection& correction,
                              const cv::Rect& inner_region, HighlightWorkspace& workspace,
                              HighlightAccumulation& accumulation, cv::cuda::Stream* stream) {
  CV_Assert(img.type() == CV_32FC3);

  const int width  = img.cols;
  const int height = img.rows;
  if (width <= 0 || height <= 0) {
    return;
  }

  workspace.Reserve(width, height);
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

  CUDA_CHECK(cudaMemsetAsync(workspace.mask_buf_, 0, static_cast<size_t>(kMaskPlanes) * pixels,
                             GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.dilated_mask_, 0,
                             static_cast<size_t>(kMaskPlanes) * pixels, GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.anyclipped_, 0, sizeof(int), GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.sums_, 0, sizeof(float) * 4, GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.cnts_, 0, sizeof(float) * 4, GetCudaStream(stream)));

  const dim3 threads(32, 8);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);

  BuildMaskKernel<<<blocks, threads, 0, cuda_stream>>>(img, workspace.mask_buf_, workspace.anyclipped_,
                                                       params);
  CUDA_CHECK(cudaGetLastError());

  int any_clipped = 0;
  CUDA_CHECK(cudaMemcpyAsync(&any_clipped, workspace.anyclipped_, sizeof(int), cudaMemcpyDeviceToHost,
                             cuda_stream));
  WaitForStream(stream);
  if (any_clipped == 0) {
    return;
  }

  DilateMaskKernel<<<blocks, threads, 0, cuda_stream>>>(workspace.mask_buf_, workspace.dilated_mask_,
                                                        width, height);
  CUDA_CHECK(cudaGetLastError());

  const cv::Rect region = NormalizeInnerRegion(inner_region, img.size());
  ChrominanceAccumulateKernel<<<blocks, threads, 0, cuda_stream>>>(
      img, workspace.dilated_mask_, workspace.sums_, workspace.cnts_, params, region.x, region.y,
      region.x + region.width, region.y + region.height);
  CUDA_CHECK(cudaGetLastError());

  std::array<float, 4> sums = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> cnts = {0.0f, 0.0f, 0.0f, 0.0f};
  CUDA_CHECK(cudaMemcpyAsync(sums.data(), workspace.sums_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  CUDA_CHECK(cudaMemcpyAsync(cnts.data(), workspace.cnts_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  WaitForStream(stream);

  accumulation.any_clipped = true;
  for (int i = 0; i < 4; ++i) {
    accumulation.sums[i] += static_cast<double>(sums[i]);
    accumulation.cnts[i] += static_cast<double>(cnts[i]);
  }
}

void ApplyHighlightCorrection(cv::cuda::GpuMat& img, const HighlightCorrection& correction,
                              HighlightWorkspace* workspace, cv::cuda::Stream* stream) {
  CV_Assert(img.type() == CV_32FC3);

  if (!correction.any_clipped) {
    Clamp01(img, stream);
    return;
  }

  HighlightWorkspace local_workspace;
  HighlightWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;
  active_workspace.Reserve(img.cols, img.rows);

  const dim3 threads(32, 8);
  const dim3 blocks((img.cols + threads.x - 1) / threads.x, (img.rows + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);

  const size_t pixels = static_cast<size_t>(img.cols) * static_cast<size_t>(img.rows);
  CUDA_CHECK(cudaMemsetAsync(active_workspace.mask_buf_, 0, static_cast<size_t>(kMaskPlanes) * pixels,
                             cuda_stream));
  CUDA_CHECK(cudaMemsetAsync(active_workspace.dilated_mask_, 0,
                             static_cast<size_t>(kMaskPlanes) * pixels, cuda_stream));

  BuildMaskKernel<<<blocks, threads, 0, cuda_stream>>>(img, active_workspace.mask_buf_,
                                                       active_workspace.anyclipped_, params);
  CUDA_CHECK(cudaGetLastError());
  DilateMaskKernel<<<blocks, threads, 0, cuda_stream>>>(active_workspace.mask_buf_,
                                                        active_workspace.dilated_mask_, img.cols,
                                                        img.rows);
  CUDA_CHECK(cudaGetLastError());

  HighlightReconstructKernel<<<blocks, threads, 0, cuda_stream>>>(
      img, active_workspace.result_, active_workspace.dilated_mask_, params);
  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    WaitForStream(stream);
  }
  img = active_workspace.result_;
}

void ApplyHighlightCorrectionAndPackRGBA(const cv::cuda::GpuMat& img, cv::cuda::GpuMat& dst,
                                         const HighlightCorrection& correction,
                                         const float* cam_mul,
                                         HighlightWorkspace* workspace,
                                         cv::cuda::Stream* stream) {
  CV_Assert(img.type() == CV_32FC3);
  if (dst.empty() || dst.size() != img.size() || dst.type() != CV_32FC4) {
    dst.create(img.size(), CV_32FC4);
  }

  HighlightWorkspace local_workspace;
  HighlightWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;
  active_workspace.Reserve(img.cols, img.rows);

  const dim3 threads(32, 8);
  const dim3 blocks((img.cols + threads.x - 1) / threads.x, (img.rows + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);
  const float3 gain      = BuildInverseCamMulScale(cam_mul);

  if (!correction.any_clipped) {
    ClampAndPackRGBAKernel<<<blocks, threads, 0, cuda_stream>>>(img, dst, gain);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
      WaitForStream(stream);
    }
    return;
  }

  const size_t pixels = static_cast<size_t>(img.cols) * static_cast<size_t>(img.rows);
  CUDA_CHECK(cudaMemsetAsync(active_workspace.mask_buf_, 0, static_cast<size_t>(kMaskPlanes) * pixels,
                             cuda_stream));
  CUDA_CHECK(cudaMemsetAsync(active_workspace.dilated_mask_, 0,
                             static_cast<size_t>(kMaskPlanes) * pixels, cuda_stream));

  BuildMaskKernel<<<blocks, threads, 0, cuda_stream>>>(img, active_workspace.mask_buf_,
                                                       active_workspace.anyclipped_, params);
  CUDA_CHECK(cudaGetLastError());
  DilateMaskKernel<<<blocks, threads, 0, cuda_stream>>>(active_workspace.mask_buf_,
                                                        active_workspace.dilated_mask_, img.cols,
                                                        img.rows);
  CUDA_CHECK(cudaGetLastError());

  HighlightReconstructAndPackRGBAKernel<<<blocks, threads, 0, cuda_stream>>>(
      img, dst, active_workspace.dilated_mask_, params, gain);
  CUDA_CHECK(cudaGetLastError());

  if (stream == nullptr) {
    WaitForStream(stream);
  }
}

void HighlightReconstruct(cv::cuda::GpuMat& img, LibRaw& raw_processor,
                          HighlightWorkspace* workspace, cv::cuda::Stream* stream) {
  HighlightCorrection correction = BuildHighlightCorrection(raw_processor);
  HighlightAccumulation accumulation;
  HighlightWorkspace local_workspace;
  HighlightWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;

  AccumulateHighlightStats(img, correction, cv::Rect{}, active_workspace, accumulation, stream);
  FinalizeHighlightCorrection(accumulation, correction);
  ApplyHighlightCorrection(img, correction, &active_workspace, stream);
}

}  // namespace CUDA
}  // namespace alcedo
