//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "decoders/processor/operators/gpu/cuda_highlight_reconstruct.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {
namespace {

constexpr float kHilightMagic = 0.987f;  // default value from darktable
constexpr float kHLPowerF     = 3.0f;
constexpr float kInvHLPowerF  = 1.0f / kHLPowerF;

__constant__ float d_correction[4];
__constant__ float d_chrominance[4];

__device__ __forceinline__ int FC(const int y, const int x) {
  // Matches CPU highlight_reconstruct.cpp's hard-coded pattern:
  // static int fc[2][2] = {{0, 1}, {2, 1}};
  return (y & 1) ? ((x & 1) ? 1 : 2) : ((x & 1) ? 1 : 0);
}

__device__ __forceinline__ int RawToCmap(const int m_width, const int row, const int col) {
  return (row / 3) * m_width + (col / 3);
}

__device__ __forceinline__ uint8_t MaskDilate(const uint8_t* in, const int w1) {
  if (in[0]) return 1;

  if (in[-w1 - 1] | in[-w1] | in[-w1 + 1] | in[-1] | in[1] | in[w1 - 1] | in[w1] | in[w1 + 1])
    return 1;

  const int w2 = 2 * w1;
  const int w3 = 3 * w1;
  return (in[-w3 - 2] | in[-w3 - 1] | in[-w3] | in[-w3 + 1] | in[-w3 + 2] | in[-w2 - 3] |
          in[-w2 - 2] | in[-w2 - 1] | in[-w2] | in[-w2 + 1] | in[-w2 + 2] | in[-w2 + 3] |
          in[-w1 - 3] | in[-w1 - 2] | in[-w1 + 2] | in[-w1 + 3] | in[-3] | in[-2] | in[2] | in[3] |
          in[w1 - 3] | in[w1 - 2] | in[w1 + 2] | in[w1 + 3] | in[w2 - 3] | in[w2 - 2] | in[w2 - 1] |
          in[w2] | in[w2 + 1] | in[w2 + 2] | in[w2 + 3] | in[w3 - 2] | in[w3 - 1] | in[w3] |
          in[w3 + 1] | in[w3 + 2])
             ? 1
             : 0;
}

__device__ __forceinline__ float CalcRefavg(const cv::cuda::PtrStep<float> in, const int row,
                                            const int col, const int height, const int width) {
  const int color = FC(row, col);
  float     mean[3] = {0.0f, 0.0f, 0.0f};
  float     cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dymin = (row > 0) ? (row - 1) : 0;
  const int dxmin = (col > 0) ? (col - 1) : 0;
  const int dymax = (row + 2 < height - 1) ? (row + 2) : (height - 1);
  const int dxmax = (col + 2 < width - 1) ? (col + 2) : (width - 1);

  for (int dy = dymin; dy < dymax; ++dy) {
    for (int dx = dxmin; dx < dxmax; ++dx) {
      const float val = fmaxf(0.0f, in(dy, dx));
      const int   c   = FC(dy, dx);
      mean[c] += val;
      cnt[c] += 1.0f;
    }
  }

  for (int c = 0; c < 3; ++c) {
    mean[c] = (cnt[c] > 0.0f) ? powf(d_correction[c] * mean[c] / cnt[c], kInvHLPowerF) : 0.0f;
  }

  const float croot_refavg[3] = {0.5f * (mean[1] + mean[2]), 0.5f * (mean[0] + mean[2]),
                                 0.5f * (mean[0] + mean[1])};
  return powf(croot_refavg[color], kHLPowerF);
}

__global__ void Clamp01Kernel(cv::cuda::PtrStep<float> img, int width, int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= width || row >= height) return;

  const float v = img(row, col);
  img(row, col) = fminf(1.0f, fmaxf(0.0f, v));
}

__global__ void BuildMaskKernel(cv::cuda::PtrStep<float> input, int width, int height,
                                uint8_t* mask_buf, int m_width, int m_height, int m_size,
                                float clip_val, int* anyclipped) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= m_width || row >= m_height) return;
  if (col < 1 || col >= m_width - 1 || row < 1 || row >= m_height - 1) return;

  char mbuff[3] = {0, 0, 0};
  const int base_raw_y = 3 * row;
  const int base_raw_x = 3 * col;

  for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
      const int raw_y = base_raw_y + y;
      const int raw_x = base_raw_x + x;

      // Mirror CPU behavior: no explicit bounds check (ranges are constructed to be safe).
      const float val     = input(raw_y, raw_x);
      const int   color   = FC(row + y, col + x);
      const char  clipped = (val >= clip_val) ? 1 : 0;
      mbuff[color] += clipped;
    }
  }

  const int idx = row * m_width + col;
  for (int c = 0; c < 3; ++c) {
    if (mbuff[c]) {
      mask_buf[c * m_size + idx] = 1;
      atomicExch(anyclipped, 1);
    }
  }
}

__global__ void DilateMaskKernel(uint8_t* mask_buf, int m_width, int m_height, int m_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= m_width || row >= m_height) return;
  if (col < 3 || col >= m_width - 3 || row < 3 || row >= m_height - 3) return;

  const int idx = row * m_width + col;
  mask_buf[3 * m_size + idx] = MaskDilate(mask_buf + 0 * m_size + idx, m_width);
  mask_buf[4 * m_size + idx] = MaskDilate(mask_buf + 1 * m_size + idx, m_width);
  mask_buf[5 * m_size + idx] = MaskDilate(mask_buf + 2 * m_size + idx, m_width);
}

__global__ void ChrominanceAccumulateKernel(cv::cuda::PtrStep<float> input, int width, int height,
                                            const uint8_t* mask_buf, int m_width, int m_size,
                                            float clip_val, float lo_clip_val, float* sums,
                                            float* cnts) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height) return;
  if (col < 3 || col >= width - 3 || row < 3 || row >= height - 3) return;

  const int   color = FC(row, col);
  const float inval = input(row, col);

  if ((inval < clip_val) && (inval > lo_clip_val) &&
      mask_buf[(color + 3) * m_size + RawToCmap(m_width, row, col)]) {
    const float ref = CalcRefavg(input, row, col, height, width);
    atomicAdd(&sums[color], inval - ref);
    atomicAdd(&cnts[color], 1.0f);
  }
}

__global__ void HighlightReconstructKernel(cv::cuda::PtrStep<float> input, cv::cuda::PtrStep<float> out,
                                           int width, int height, float clip_val) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height) return;

  const int   color = FC(row, col);
  const float inval = fmaxf(0.0f, input(row, col));

  if (inval >= clip_val) {
    const float ref = CalcRefavg(input, row, col, height, width);
    out(row, col)   = fmaxf(inval, ref + d_chrominance[color]);
  } else {
    out(row, col) = inval;
  }
}

static int RoundSize(const int size, const int alignment) {
  return ((size % alignment) == 0) ? size : ((size - 1) / alignment + 1) * alignment;
}

}  // namespace

void Clamp01(cv::cuda::GpuMat& img) {
  CV_Assert(img.type() == CV_32FC1);

  const dim3 threads(32, 32);
  const dim3 blocks((img.cols + threads.x - 1) / threads.x, (img.rows + threads.y - 1) / threads.y);
  Clamp01Kernel<<<blocks, threads>>>(img, img.cols, img.rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void HighlightReconstruct(cv::cuda::GpuMat& img, LibRaw& raw_processor) {
  CV_Assert(img.type() == CV_32FC1);

  const int width  = img.cols;
  const int height = img.rows;

  const float* cam_mul = raw_processor.imgdata.color.cam_mul;
  float        correction[4] = {cam_mul[0] / cam_mul[1], 1.0f, cam_mul[2] / cam_mul[1], 0.0f};
  CUDA_CHECK(cudaMemcpyToSymbol(d_correction, correction, sizeof(float) * 4));

  const float clip_val = kHilightMagic;

  const int   m_width  = width / 3;
  const int   m_height = height / 3;
  const int   m_size   = RoundSize((m_width + 1) * (m_height + 1), 16);

  uint8_t*    d_mask_buf = nullptr;
  int*        d_anyclipped = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mask_buf, static_cast<size_t>(6) * m_size * sizeof(uint8_t)));
  CUDA_CHECK(cudaMemset(d_mask_buf, 0, static_cast<size_t>(6) * m_size * sizeof(uint8_t)));

  CUDA_CHECK(cudaMalloc(&d_anyclipped, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_anyclipped, 0, sizeof(int)));

  {
    const dim3 threads(32, 32);
    const dim3 blocks((m_width + threads.x - 1) / threads.x, (m_height + threads.y - 1) / threads.y);
    BuildMaskKernel<<<blocks, threads>>>(img, width, height, d_mask_buf, m_width, m_height, m_size,
                                         clip_val, d_anyclipped);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  int anyclipped = 0;
  CUDA_CHECK(cudaMemcpy(&anyclipped, d_anyclipped, sizeof(int), cudaMemcpyDeviceToHost));
  if (!anyclipped) {
    CUDA_CHECK(cudaFree(d_anyclipped));
    CUDA_CHECK(cudaFree(d_mask_buf));
    return;
  }

  {
    const dim3 threads(32, 32);
    const dim3 blocks((m_width + threads.x - 1) / threads.x, (m_height + threads.y - 1) / threads.y);
    DilateMaskKernel<<<blocks, threads>>>(d_mask_buf, m_width, m_height, m_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float* d_sums = nullptr;
  float* d_cnts = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sums, sizeof(float) * 4));
  CUDA_CHECK(cudaMalloc(&d_cnts, sizeof(float) * 4));
  CUDA_CHECK(cudaMemset(d_sums, 0, sizeof(float) * 4));
  CUDA_CHECK(cudaMemset(d_cnts, 0, sizeof(float) * 4));

  {
    const float lo_clip_val = 0.99f * clip_val;
    const dim3  threads(32, 32);
    const dim3  blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    ChrominanceAccumulateKernel<<<blocks, threads>>>(img, width, height, d_mask_buf, m_width, m_size,
                                                     clip_val, lo_clip_val, d_sums, d_cnts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::array<float, 4> sums = {0.f, 0.f, 0.f, 0.f};
  std::array<float, 4> cnts = {0.f, 0.f, 0.f, 0.f};
  CUDA_CHECK(cudaMemcpy(sums.data(), d_sums, sizeof(float) * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cnts.data(), d_cnts, sizeof(float) * 4, cudaMemcpyDeviceToHost));

  float chrominance[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < 3; ++c) {
    chrominance[c] = (cnts[c] > 80.0f) ? (sums[c] / cnts[c]) : 0.0f;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(d_chrominance, chrominance, sizeof(float) * 4));

  cv::cuda::GpuMat result;
  result.create(img.size(), img.type());

  {
    const dim3 threads(32, 32);
    const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    HighlightReconstructKernel<<<blocks, threads>>>(img, result, width, height, clip_val);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  img = result;

  CUDA_CHECK(cudaFree(d_cnts));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFree(d_anyclipped));
  CUDA_CHECK(cudaFree(d_mask_buf));
}

}  // namespace CUDA
}  // namespace puerhlab
