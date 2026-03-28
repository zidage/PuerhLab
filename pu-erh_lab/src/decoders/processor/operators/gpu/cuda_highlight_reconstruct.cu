//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
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

  mask_buf[kPlaneBaseR * input.rows * input.cols + idx] = pixel.x >= params.clips[0] ? 1 : 0;
  mask_buf[kPlaneBaseG * input.rows * input.cols + idx] = pixel.y >= params.clips[1] ? 1 : 0;
  mask_buf[kPlaneBaseB * input.rows * input.cols + idx] = pixel.z >= params.clips[2] ? 1 : 0;
  mask_buf[kPlaneBaseMulti * input.rows * input.cols + idx] = count >= 2 ? 1 : 0;

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
                                            float* cnts, HighlightCorrectionParams params) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
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

  const int    size  = input.rows * input.cols;
  const float3 pixel = MaxRgb(input.ptr(row)[col]);
  const int    count = CountClippedChannels(pixel, params.clips);

  if (count == 0) {
    output.ptr(row)[col] = pixel;
    return;
  }

  float3 result = pixel;
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

  // if (count >= 2) {
  //   const float neutral =
  //       LocalNeutralTarget(input, dilated_mask_buf + kPlaneDilMulti * size, row, col);
  //   if (count == 3) {
  //     result = make_float3(neutral, neutral, neutral);
  //   } else {
  //     constexpr float kDesatBlend = 0.5f;
  //     result.x = result.x * (1.0f - kDesatBlend) + neutral * kDesatBlend;
  //     result.y = result.y * (1.0f - kDesatBlend) + neutral * kDesatBlend;
  //     result.z = result.z * (1.0f - kDesatBlend) + neutral * kDesatBlend;
  //   }
  // }

  output.ptr(row)[col] = result;
}

}  // namespace

void Clamp01(cv::cuda::GpuMat& img) {
  if (img.type() != CV_32FC1 && img.type() != CV_32FC3) {
    throw std::runtime_error("CUDA::Clamp01: only CV_32FC1/CV_32FC3 are supported");
  }

  const dim3 threads(32, 8);
  const dim3 blocks((img.cols + threads.x - 1) / threads.x, (img.rows + threads.y - 1) / threads.y);

  if (img.type() == CV_32FC1) {
    Clamp01KernelGray<<<blocks, threads>>>(img, img.cols, img.rows);
  } else {
    Clamp01KernelRgb<<<blocks, threads>>>(img);
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void HighlightReconstruct(cv::cuda::GpuMat& img, LibRaw& raw_processor) {
  CV_Assert(img.type() == CV_32FC3);

  const int width  = img.cols;
  const int height = img.rows;
  if (width == 0 || height == 0) {
    return;
  }

  const float* cam_mul = raw_processor.imgdata.color.cam_mul;
  const float  green   = std::max(cam_mul[1], 1e-6f);

  HighlightCorrectionParams params = {};
  params.clips[0]                  = kHilightMagic * (cam_mul[0] / green);
  params.clips[1]                  = kHilightMagic;
  params.clips[2]                  = kHilightMagic * (cam_mul[2] / green);
  params.clipdark[0]               = 0.03f * params.clips[0];
  params.clipdark[1]               = 0.125f * params.clips[1];
  params.clipdark[2]               = 0.03f * params.clips[2];

  const int size = width * height;

  uint8_t* d_mask_buf     = nullptr;
  uint8_t* d_dilated_mask = nullptr;
  int*     d_anyclipped   = nullptr;
  float*   d_sums         = nullptr;
  float*   d_cnts         = nullptr;

  CUDA_CHECK(cudaMalloc(&d_mask_buf, static_cast<size_t>(kMaskPlanes) * size * sizeof(uint8_t)));
  CUDA_CHECK(
      cudaMalloc(&d_dilated_mask, static_cast<size_t>(kMaskPlanes) * size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMemset(d_mask_buf, 0, static_cast<size_t>(kMaskPlanes) * size * sizeof(uint8_t)));
  CUDA_CHECK(
      cudaMemset(d_dilated_mask, 0, static_cast<size_t>(kMaskPlanes) * size * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc(&d_anyclipped, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_anyclipped, 0, sizeof(int)));

  const dim3 threads(32, 8);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  BuildMaskKernel<<<blocks, threads>>>(img, d_mask_buf, d_anyclipped, params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int anyclipped = 0;
  CUDA_CHECK(cudaMemcpy(&anyclipped, d_anyclipped, sizeof(int), cudaMemcpyDeviceToHost));
  if (!anyclipped) {
    CUDA_CHECK(cudaFree(d_anyclipped));
    CUDA_CHECK(cudaFree(d_dilated_mask));
    CUDA_CHECK(cudaFree(d_mask_buf));
    return;
  }

  DilateMaskKernel<<<blocks, threads>>>(d_mask_buf, d_dilated_mask, width, height);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMalloc(&d_sums, sizeof(float) * 4));
  CUDA_CHECK(cudaMalloc(&d_cnts, sizeof(float) * 4));
  CUDA_CHECK(cudaMemset(d_sums, 0, sizeof(float) * 4));
  CUDA_CHECK(cudaMemset(d_cnts, 0, sizeof(float) * 4));

  ChrominanceAccumulateKernel<<<blocks, threads>>>(img, d_dilated_mask, d_sums, d_cnts, params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::array<float, 4> sums = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> cnts = {0.0f, 0.0f, 0.0f, 0.0f};
  CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, sizeof(float) * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cnts.data(), d_cnts, sizeof(float) * 4, cudaMemcpyDeviceToHost));

  for (int c = 0; c < 3; ++c) {
    params.chrominance[c] = (cnts[c] > 0.0f) ? (sums[c] / cnts[c]) : 0.0f;
  }

  cv::cuda::GpuMat result(img.size(), img.type());
  HighlightReconstructKernel<<<blocks, threads>>>(img, result, d_dilated_mask, params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  img = result;

  CUDA_CHECK(cudaFree(d_cnts));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFree(d_anyclipped));
  CUDA_CHECK(cudaFree(d_dilated_mask));
  CUDA_CHECK(cudaFree(d_mask_buf));
}

}  // namespace CUDA
}  // namespace puerhlab
