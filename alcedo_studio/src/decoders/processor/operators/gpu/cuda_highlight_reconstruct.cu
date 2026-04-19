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
constexpr int   kStatsBlockX    = 32;
constexpr int   kStatsBlockY    = 8;
constexpr int   kStatsTileW     = kStatsBlockX + 2 * kDilateRadius;
constexpr int   kStatsTileH     = kStatsBlockY + 2 * kDilateRadius;

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

auto GetOrientedOutputSize(const cv::Size& size, const int flip) -> cv::Size {
  switch (flip) {
    case 5:
    case 6:
      return {size.height, size.width};
    default:
      return size;
  }
}

__device__ __forceinline__ float Cube(const float value) { return value * value * value; }

__device__ __forceinline__ float3 ClampRgb(const float3& value) {
  return make_float3(fminf(1.0f, fmaxf(0.0f, value.x)), fminf(1.0f, fmaxf(0.0f, value.y)),
                     fminf(1.0f, fmaxf(0.0f, value.z)));
}

__device__ __forceinline__ float3 MaxRgb(const float3& value) {
  return make_float3(fmaxf(0.0f, value.x), fmaxf(0.0f, value.y), fmaxf(0.0f, value.z));
}

__device__ __forceinline__ float3 LoadPlanarRgb(const cv::cuda::PtrStepSz<float> red,
                                                const cv::cuda::PtrStepSz<float> green,
                                                const cv::cuda::PtrStepSz<float> blue,
                                                const int row, const int col) {
  return make_float3(red.ptr(row)[col], green.ptr(row)[col], blue.ptr(row)[col]);
}

__device__ __forceinline__ void StoreOrientedRgba(cv::cuda::PtrStepSz<float4> output,
                                                  const int src_row, const int src_col,
                                                  const int src_rows, const int src_cols,
                                                  const float4 value, const int flip) {
  switch (flip) {
    case 3:
      output.ptr(src_rows - 1 - src_row)[src_cols - 1 - src_col] = value;
      break;
    case 5:
      output.ptr(src_cols - 1 - src_col)[src_row] = value;
      break;
    case 6:
      output.ptr(src_col)[src_rows - 1 - src_row] = value;
      break;
    default:
      output.ptr(src_row)[src_col] = value;
      break;
  }
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

__device__ __forceinline__ float3 CalcRefavgPlanar(const cv::cuda::PtrStepSz<float> red,
                                                   const cv::cuda::PtrStepSz<float> green,
                                                   const cv::cuda::PtrStepSz<float> blue,
                                                   const int row, const int col) {
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dymin = max(0, row - 1);
  const int dxmin = max(0, col - 1);
  const int dymax = min(red.rows - 1, row + 1);
  const int dxmax = min(red.cols - 1, col + 1);

  for (int dy = dymin; dy <= dymax; ++dy) {
    for (int dx = dxmin; dx <= dxmax; ++dx) {
      const float3 sample = MaxRgb(LoadPlanarRgb(red, green, blue, dy, dx));
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

__device__ __forceinline__ int CountClippedChannels(const float3& pixel, const float* clips) {
  int count = 0;
  count += pixel.x >= clips[0] ? 1 : 0;
  count += pixel.y >= clips[1] ? 1 : 0;
  count += pixel.z >= clips[2] ? 1 : 0;
  return count;
}

template <typename Tile>
__device__ __forceinline__ float3 CalcRefavgFromTile(const Tile& tile, const int row, const int col,
                                                     const int local_y, const int local_x,
                                                     const int height, const int width) {
  float mean[3] = {0.0f, 0.0f, 0.0f};
  float cnt[3]  = {0.0f, 0.0f, 0.0f};

  const int dy0 = max(-1, -row);
  const int dx0 = max(-1, -col);
  const int dy1 = min(1, height - 1 - row);
  const int dx1 = min(1, width - 1 - col);

  for (int dy = dy0; dy <= dy1; ++dy) {
    for (int dx = dx0; dx <= dx1; ++dx) {
      const float3 sample = tile[local_y + dy][local_x + dx];
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

__global__ void AccumulateHighlightStatsKernel(const cv::cuda::PtrStepSz<float3> input,
                                               int* anyclipped, float* sums, float* cnts,
                                               HighlightCorrectionParams params, const int x0,
                                               const int y0, const int x1, const int y1) {
  __shared__ float3  tile_img[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_r[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_g[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_b[kStatsTileH][kStatsTileW];
  __shared__ float   block_sums[3];
  __shared__ float   block_cnts[3];
  __shared__ int     block_any_clipped;

  const int lane = threadIdx.y * blockDim.x + threadIdx.x;
  if (lane < 3) {
    block_sums[lane] = 0.0f;
    block_cnts[lane] = 0.0f;
  }
  if (lane == 0) {
    block_any_clipped = 0;
  }

  const int tile_origin_x = blockIdx.x * blockDim.x - kDilateRadius;
  const int tile_origin_y = blockIdx.y * blockDim.y - kDilateRadius;
  for (int sy = threadIdx.y; sy < kStatsTileH; sy += blockDim.y) {
    const int gy = max(0, min(input.rows - 1, tile_origin_y + sy));
    for (int sx = threadIdx.x; sx < kStatsTileW; sx += blockDim.x) {
      const int   gx    = max(0, min(input.cols - 1, tile_origin_x + sx));
      const float3 pixel = MaxRgb(input.ptr(gy)[gx]);
      tile_img[sy][sx]   = pixel;
      tile_r[sy][sx]     = pixel.x >= params.clips[0] ? 1 : 0;
      tile_g[sy][sx]     = pixel.y >= params.clips[1] ? 1 : 0;
      tile_b[sy][sx]     = pixel.z >= params.clips[2] ? 1 : 0;

      if ((tile_r[sy][sx] | tile_g[sy][sx] | tile_b[sy][sx]) != 0) {
        atomicExch(&block_any_clipped, 1);
      }
    }
  }
  __syncthreads();

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const bool in_bounds = col < input.cols && row < input.rows;
  if (in_bounds && col >= x0 && col < x1 && row >= y0 && row < y1) {
    const int local_x = threadIdx.x + kDilateRadius;
    const int local_y = threadIdx.y + kDilateRadius;
    uint8_t   dil_r   = 0;
    uint8_t   dil_g   = 0;
    uint8_t   dil_b   = 0;

    const int dy0 = max(-kDilateRadius, -row);
    const int dx0 = max(-kDilateRadius, -col);
    const int dy1 = min(kDilateRadius, input.rows - 1 - row);
    const int dx1 = min(kDilateRadius, input.cols - 1 - col);

    #pragma unroll
    for (int dy = -kDilateRadius; dy <= kDilateRadius; ++dy) {
      if (dy < dy0 || dy > dy1) {
        continue;
      }
      #pragma unroll
      for (int dx = -kDilateRadius; dx <= kDilateRadius; ++dx) {
        if (dx < dx0 || dx > dx1) {
          continue;
        }
        dil_r |= tile_r[local_y + dy][local_x + dx];
        dil_g |= tile_g[local_y + dy][local_x + dx];
        dil_b |= tile_b[local_y + dy][local_x + dx];
      }
    }

    const float3 pixel = tile_img[local_y][local_x];
    const bool   use_r = dil_r != 0 && pixel.x > params.clipdark[0] && pixel.x < params.clips[0];
    const bool   use_g = dil_g != 0 && pixel.y > params.clipdark[1] && pixel.y < params.clips[1];
    const bool   use_b = dil_b != 0 && pixel.z > params.clipdark[2] && pixel.z < params.clips[2];

    if (use_r || use_g || use_b) {
      const float3 ref =
          CalcRefavgFromTile(tile_img, row, col, local_y, local_x, input.rows, input.cols);
      if (use_r) {
        atomicAdd(&block_sums[0], pixel.x - ref.x);
        atomicAdd(&block_cnts[0], 1.0f);
      }
      if (use_g) {
        atomicAdd(&block_sums[1], pixel.y - ref.y);
        atomicAdd(&block_cnts[1], 1.0f);
      }
      if (use_b) {
        atomicAdd(&block_sums[2], pixel.z - ref.z);
        atomicAdd(&block_cnts[2], 1.0f);
      }
    }
  }

  __syncthreads();
  if (lane == 0 && block_any_clipped != 0) {
    atomicExch(anyclipped, 1);
  }
  if (lane < 3 && (block_sums[lane] != 0.0f || block_cnts[lane] != 0.0f)) {
    atomicAdd(&sums[lane], block_sums[lane]);
    atomicAdd(&cnts[lane], block_cnts[lane]);
  }
}

__global__ void AccumulateHighlightStatsPlanarKernel(const cv::cuda::PtrStepSz<float> red,
                                                     const cv::cuda::PtrStepSz<float> green,
                                                     const cv::cuda::PtrStepSz<float> blue,
                                                     int* anyclipped, float* sums, float* cnts,
                                                     HighlightCorrectionParams params, const int x0,
                                                     const int y0, const int x1, const int y1) {
  __shared__ float3  tile_img[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_r[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_g[kStatsTileH][kStatsTileW];
  __shared__ uint8_t tile_b[kStatsTileH][kStatsTileW];
  __shared__ float   block_sums[3];
  __shared__ float   block_cnts[3];
  __shared__ int     block_any_clipped;

  const int lane = threadIdx.y * blockDim.x + threadIdx.x;
  if (lane < 3) {
    block_sums[lane] = 0.0f;
    block_cnts[lane] = 0.0f;
  }
  if (lane == 0) {
    block_any_clipped = 0;
  }

  const int tile_origin_x = blockIdx.x * blockDim.x - kDilateRadius;
  const int tile_origin_y = blockIdx.y * blockDim.y - kDilateRadius;
  for (int sy = threadIdx.y; sy < kStatsTileH; sy += blockDim.y) {
    const int gy = max(0, min(red.rows - 1, tile_origin_y + sy));
    for (int sx = threadIdx.x; sx < kStatsTileW; sx += blockDim.x) {
      const int gx = max(0, min(red.cols - 1, tile_origin_x + sx));
      const float3 pixel = MaxRgb(LoadPlanarRgb(red, green, blue, gy, gx));
      tile_img[sy][sx]   = pixel;
      tile_r[sy][sx]     = pixel.x >= params.clips[0] ? 1 : 0;
      tile_g[sy][sx]     = pixel.y >= params.clips[1] ? 1 : 0;
      tile_b[sy][sx]     = pixel.z >= params.clips[2] ? 1 : 0;

      if ((tile_r[sy][sx] | tile_g[sy][sx] | tile_b[sy][sx]) != 0) {
        atomicExch(&block_any_clipped, 1);
      }
    }
  }
  __syncthreads();

  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const bool in_bounds = col < red.cols && row < red.rows;
  if (in_bounds && col >= x0 && col < x1 && row >= y0 && row < y1) {
    const int local_x = threadIdx.x + kDilateRadius;
    const int local_y = threadIdx.y + kDilateRadius;
    uint8_t   dil_r   = 0;
    uint8_t   dil_g   = 0;
    uint8_t   dil_b   = 0;

    const int dy0 = max(-kDilateRadius, -row);
    const int dx0 = max(-kDilateRadius, -col);
    const int dy1 = min(kDilateRadius, red.rows - 1 - row);
    const int dx1 = min(kDilateRadius, red.cols - 1 - col);

    #pragma unroll
    for (int dy = -kDilateRadius; dy <= kDilateRadius; ++dy) {
      if (dy < dy0 || dy > dy1) {
        continue;
      }
      #pragma unroll
      for (int dx = -kDilateRadius; dx <= kDilateRadius; ++dx) {
        if (dx < dx0 || dx > dx1) {
          continue;
        }
        dil_r |= tile_r[local_y + dy][local_x + dx];
        dil_g |= tile_g[local_y + dy][local_x + dx];
        dil_b |= tile_b[local_y + dy][local_x + dx];
      }
    }

    const float3 pixel = tile_img[local_y][local_x];
    const bool   use_r = dil_r != 0 && pixel.x > params.clipdark[0] && pixel.x < params.clips[0];
    const bool   use_g = dil_g != 0 && pixel.y > params.clipdark[1] && pixel.y < params.clips[1];
    const bool   use_b = dil_b != 0 && pixel.z > params.clipdark[2] && pixel.z < params.clips[2];

    if (use_r || use_g || use_b) {
      const float3 ref =
          CalcRefavgFromTile(tile_img, row, col, local_y, local_x, red.rows, red.cols);
      if (use_r) {
        atomicAdd(&block_sums[0], pixel.x - ref.x);
        atomicAdd(&block_cnts[0], 1.0f);
      }
      if (use_g) {
        atomicAdd(&block_sums[1], pixel.y - ref.y);
        atomicAdd(&block_cnts[1], 1.0f);
      }
      if (use_b) {
        atomicAdd(&block_sums[2], pixel.z - ref.z);
        atomicAdd(&block_cnts[2], 1.0f);
      }
    }
  }

  __syncthreads();
  if (lane == 0 && block_any_clipped != 0) {
    atomicExch(anyclipped, 1);
  }
  if (lane < 3 && (block_sums[lane] != 0.0f || block_cnts[lane] != 0.0f)) {
    atomicAdd(&sums[lane], block_sums[lane]);
    atomicAdd(&cnts[lane], block_cnts[lane]);
  }
}

__global__ void HighlightReconstructKernel(const cv::cuda::PtrStepSz<float3> input,
                                           cv::cuda::PtrStepSz<float3> output,
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
                                       cv::cuda::PtrStepSz<float4> output, const float3 gain,
                                       const int flip) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= input.cols || row >= input.rows) {
    return;
  }

  const float3 pixel = ClampRgb(input.ptr(row)[col]);
  StoreOrientedRgba(output, row, col, input.rows, input.cols,
                    make_float4(pixel.x * gain.x, pixel.y * gain.y, pixel.z * gain.z, 1.0f),
                    flip);
}

__global__ void ClampAndPackRGBAPlanarKernel(const cv::cuda::PtrStepSz<float> red,
                                             const cv::cuda::PtrStepSz<float> green,
                                             const cv::cuda::PtrStepSz<float> blue,
                                             cv::cuda::PtrStepSz<float4> output,
                                             const float3 gain, const int flip) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= red.cols || row >= red.rows) {
    return;
  }

  const float3 pixel = ClampRgb(LoadPlanarRgb(red, green, blue, row, col));
  StoreOrientedRgba(output, row, col, red.rows, red.cols,
                    make_float4(pixel.x * gain.x, pixel.y * gain.y, pixel.z * gain.z, 1.0f),
                    flip);
}

__global__ void HighlightReconstructAndPackRGBAKernel(const cv::cuda::PtrStepSz<float3> input,
                                                      cv::cuda::PtrStepSz<float4> output,
                                                      HighlightCorrectionParams params,
                                                      const float3 gain, const int flip) {
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
  StoreOrientedRgba(output, row, col, input.rows, input.cols,
                    make_float4(result.x * gain.x, result.y * gain.y, result.z * gain.z, 1.0f),
                    flip);
}

__global__ void HighlightReconstructAndPackRGBAPlanarKernel(
    const cv::cuda::PtrStepSz<float> red, const cv::cuda::PtrStepSz<float> green,
    const cv::cuda::PtrStepSz<float> blue, cv::cuda::PtrStepSz<float4> output,
    HighlightCorrectionParams params, const float3 gain, const int flip) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= red.cols || row >= red.rows) {
    return;
  }

  const float3 pixel = MaxRgb(LoadPlanarRgb(red, green, blue, row, col));
  const int    count = CountClippedChannels(pixel, params.clips);

  float3 result = pixel;
  if (count != 0) {
    const float3 ref = CalcRefavgPlanar(red, green, blue, row, col);
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
  StoreOrientedRgba(output, row, col, red.rows, red.cols,
                    make_float4(result.x * gain.x, result.y * gain.y, result.z * gain.z, 1.0f),
                    flip);
}

}  // namespace

HighlightWorkspace::HighlightWorkspace() = default;

HighlightWorkspace::~HighlightWorkspace() { Release(); }

HighlightWorkspace::HighlightWorkspace(HighlightWorkspace&& other) noexcept
    : anyclipped_(std::exchange(other.anyclipped_, nullptr)),
      sums_(std::exchange(other.sums_, nullptr)),
      cnts_(std::exchange(other.cnts_, nullptr)),
      mask_capacity_(std::exchange(other.mask_capacity_, 0)),
      result_(std::move(other.result_)) {}

auto HighlightWorkspace::operator=(HighlightWorkspace&& other) noexcept -> HighlightWorkspace& {
  if (this != &other) {
    Release();
    anyclipped_ = std::exchange(other.anyclipped_, nullptr);
    sums_       = std::exchange(other.sums_, nullptr);
    cnts_       = std::exchange(other.cnts_, nullptr);
    mask_capacity_ = std::exchange(other.mask_capacity_, 0);
    result_      = std::move(other.result_);
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
  CUDA_CHECK(cudaMemsetAsync(workspace.anyclipped_, 0, sizeof(int), GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.sums_, 0, sizeof(float) * 4, GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.cnts_, 0, sizeof(float) * 4, GetCudaStream(stream)));

  const dim3 threads(kStatsBlockX, kStatsBlockY);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);

  const cv::Rect region = NormalizeInnerRegion(inner_region, img.size());
  AccumulateHighlightStatsKernel<<<blocks, threads, 0, cuda_stream>>>(
      img, workspace.anyclipped_, workspace.sums_, workspace.cnts_, params,
      region.x, region.y, region.x + region.width, region.y + region.height);
  CUDA_CHECK(cudaGetLastError());

  int any_clipped = 0;
  std::array<float, 4> sums = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> cnts = {0.0f, 0.0f, 0.0f, 0.0f};
  CUDA_CHECK(cudaMemcpyAsync(&any_clipped, workspace.anyclipped_, sizeof(int), cudaMemcpyDeviceToHost,
                             cuda_stream));
  CUDA_CHECK(cudaMemcpyAsync(sums.data(), workspace.sums_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  CUDA_CHECK(cudaMemcpyAsync(cnts.data(), workspace.cnts_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  WaitForStream(stream);

  if (any_clipped == 0) {
    return;
  }
  accumulation.any_clipped = true;
  for (int i = 0; i < 4; ++i) {
    accumulation.sums[i] += static_cast<double>(sums[i]);
    accumulation.cnts[i] += static_cast<double>(cnts[i]);
  }
}

void AccumulateHighlightStats(const cv::cuda::GpuMat& red, const cv::cuda::GpuMat& green,
                              const cv::cuda::GpuMat& blue,
                              const HighlightCorrection& correction,
                              const cv::Rect& inner_region, HighlightWorkspace& workspace,
                              HighlightAccumulation& accumulation, cv::cuda::Stream* stream) {
  CV_Assert(red.type() == CV_32FC1 && green.type() == CV_32FC1 && blue.type() == CV_32FC1);
  CV_Assert(red.size() == green.size() && red.size() == blue.size());

  const int width  = red.cols;
  const int height = red.rows;
  if (width <= 0 || height <= 0) {
    return;
  }

  workspace.Reserve(width, height);
  CUDA_CHECK(cudaMemsetAsync(workspace.anyclipped_, 0, sizeof(int), GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.sums_, 0, sizeof(float) * 4, GetCudaStream(stream)));
  CUDA_CHECK(cudaMemsetAsync(workspace.cnts_, 0, sizeof(float) * 4, GetCudaStream(stream)));

  const dim3 threads(kStatsBlockX, kStatsBlockY);
  const dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);

  const cv::Rect region = NormalizeInnerRegion(inner_region, red.size());
  AccumulateHighlightStatsPlanarKernel<<<blocks, threads, 0, cuda_stream>>>(
      red, green, blue, workspace.anyclipped_, workspace.sums_, workspace.cnts_, params,
      region.x, region.y, region.x + region.width, region.y + region.height);
  CUDA_CHECK(cudaGetLastError());

  int any_clipped = 0;
  std::array<float, 4> sums = {0.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 4> cnts = {0.0f, 0.0f, 0.0f, 0.0f};
  CUDA_CHECK(cudaMemcpyAsync(&any_clipped, workspace.anyclipped_, sizeof(int), cudaMemcpyDeviceToHost,
                             cuda_stream));
  CUDA_CHECK(cudaMemcpyAsync(sums.data(), workspace.sums_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  CUDA_CHECK(cudaMemcpyAsync(cnts.data(), workspace.cnts_, sizeof(float) * 4, cudaMemcpyDeviceToHost,
                             cuda_stream));
  WaitForStream(stream);

  if (any_clipped == 0) {
    return;
  }
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

  HighlightReconstructKernel<<<blocks, threads, 0, cuda_stream>>>(img, active_workspace.result_,
                                                                  params);
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
  ApplyHighlightCorrectionAndPackRGBAOriented(img, dst, correction, cam_mul, 0, workspace, stream);
}

void ApplyHighlightCorrectionAndPackRGBA(const cv::cuda::GpuMat& red,
                                         const cv::cuda::GpuMat& green,
                                         const cv::cuda::GpuMat& blue,
                                         cv::cuda::GpuMat& dst,
                                         const HighlightCorrection& correction,
                                         const float* cam_mul,
                                         HighlightWorkspace* workspace,
                                         cv::cuda::Stream* stream) {
  ApplyHighlightCorrectionAndPackRGBAOriented(red, green, blue, dst, correction, cam_mul, 0,
                                              workspace, stream);
}

void ApplyHighlightCorrectionAndPackRGBAOriented(const cv::cuda::GpuMat& img, cv::cuda::GpuMat& dst,
                                                 const HighlightCorrection& correction,
                                                 const float* cam_mul, const int flip,
                                                 HighlightWorkspace* workspace,
                                                 cv::cuda::Stream* stream) {
  CV_Assert(img.type() == CV_32FC3);
  const cv::Size dst_size = GetOrientedOutputSize(img.size(), flip);
  if (dst.empty() || dst.size() != dst_size || dst.type() != CV_32FC4) {
    dst.create(dst_size, CV_32FC4);
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
    ClampAndPackRGBAKernel<<<blocks, threads, 0, cuda_stream>>>(img, dst, gain, flip);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
      WaitForStream(stream);
    }
    return;
  }

  HighlightReconstructAndPackRGBAKernel<<<blocks, threads, 0, cuda_stream>>>(img, dst, params, gain,
                                                                              flip);
  CUDA_CHECK(cudaGetLastError());

  if (stream == nullptr) {
    WaitForStream(stream);
  }
}

void ApplyHighlightCorrectionAndPackRGBAOriented(const cv::cuda::GpuMat& red,
                                                 const cv::cuda::GpuMat& green,
                                                 const cv::cuda::GpuMat& blue,
                                                 cv::cuda::GpuMat& dst,
                                                 const HighlightCorrection& correction,
                                                 const float* cam_mul, const int flip,
                                                 HighlightWorkspace* workspace,
                                                 cv::cuda::Stream* stream) {
  CV_Assert(red.type() == CV_32FC1 && green.type() == CV_32FC1 && blue.type() == CV_32FC1);
  CV_Assert(red.size() == green.size() && red.size() == blue.size());

  const cv::Size dst_size = GetOrientedOutputSize(red.size(), flip);
  if (dst.empty() || dst.size() != dst_size || dst.type() != CV_32FC4) {
    dst.create(dst_size, CV_32FC4);
  }

  HighlightWorkspace local_workspace;
  HighlightWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;
  active_workspace.Reserve(red.cols, red.rows);

  const dim3 threads(32, 8);
  const dim3 blocks((red.cols + threads.x - 1) / threads.x, (red.rows + threads.y - 1) / threads.y);
  const auto params      = ToParams(correction);
  const auto cuda_stream = GetCudaStream(stream);
  const float3 gain      = BuildInverseCamMulScale(cam_mul);

  if (!correction.any_clipped) {
    ClampAndPackRGBAPlanarKernel<<<blocks, threads, 0, cuda_stream>>>(red, green, blue, dst, gain,
                                                                      flip);
    CUDA_CHECK(cudaGetLastError());
    if (stream == nullptr) {
      WaitForStream(stream);
    }
    return;
  }

  HighlightReconstructAndPackRGBAPlanarKernel<<<blocks, threads, 0, cuda_stream>>>(
      red, green, blue, dst, params, gain, flip);
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
