//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//
//  This file also contains material subject to the upstream notices below.

/**
* RATIO CORRECTED DEMOSAICING
* Luis Sanz Rodríguez (luis.sanz.rodriguez(at)gmail(dot)com)
*
* Release 2.3 @ 171125
*/

#include "decoders/processor/operators/gpu/cuda_debayer_rcd.hpp"
#include "decoders/processor/operators/gpu/cuda_image_ops.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {

namespace {

constexpr float kEps              = 1e-5f;
constexpr float kEpsSq            = 1e-10f;
constexpr int   kRcdRadius        = 4;
constexpr int   kEdgeFallbackRadius = kRcdRadius;
constexpr int   kThreadsX         = 32;
constexpr int   kThreadsY         = 8;
constexpr int   kTileWidth        = kThreadsX + 2 * kRcdRadius;
constexpr int   kTileHeight       = kThreadsY + 2 * kRcdRadius;

__device__ __forceinline__ int FC(const BayerPattern2x2& pattern, const int y, const int x) {
  return pattern.rgb_fc[BayerCellIndex(y, x)];
}

__device__ __forceinline__ int ClampCoord(const int value, const int limit) {
  return max(0, min(value, limit - 1));
}

__device__ __forceinline__ float SafeRawRead(const cv::cuda::PtrStep<float> raw, const int width,
                                             const int height, const int y, const int x) {
  return raw.ptr(ClampCoord(y, height))[ClampCoord(x, width)];
}

__device__ __forceinline__ void LoadRawTileFull(const cv::cuda::PtrStep<float> raw, const int width,
                                                const int height,
                                                float tile[kTileHeight][kTileWidth]) {
  const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - kRcdRadius;
  const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - kRcdRadius;

  for (int tile_y = threadIdx.y; tile_y < kTileHeight; tile_y += blockDim.y) {
    const int    gy       = ClampCoord(tile_y0 + tile_y, height);
    const float* raw_row  = raw.ptr(gy);
    float*       tile_row = tile[tile_y];
    for (int tile_x = threadIdx.x; tile_x < kTileWidth; tile_x += blockDim.x) {
      tile_row[tile_x] = raw_row[ClampCoord(tile_x0 + tile_x, width)];
    }
  }
}

__device__ __forceinline__ float LowPassAtTile(const float tile[kTileHeight][kTileWidth],
                                                const int y, const int x) {
  const float c  = tile[y][x];
  const float n  = tile[y - 1][x];
  const float s  = tile[y + 1][x];
  const float w  = tile[y][x - 1];
  const float e  = tile[y][x + 1];

  const float nw = tile[y - 1][x - 1];
  const float ne = tile[y - 1][x + 1];
  const float sw = tile[y + 1][x - 1];
  const float se = tile[y + 1][x + 1];

  return 0.25f * c + 0.125f * (n + s + w + e) + 0.0625f * (nw + ne + sw + se);
}

__device__ float EstimateEdgeChannel(const cv::cuda::PtrStep<float> raw, const int width,
                                     const int height, const BayerPattern2x2& pattern, const int y,
                                     const int x, const int target_color) {
  if (FC(pattern, y, x) == target_color) {
    return raw.ptr(y)[x];
  }

  float weighted_sum = 0.0f;
  float weight_sum   = 0.0f;

  for (int radius = 1; radius <= kEdgeFallbackRadius; ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (max(abs(dx), abs(dy)) != radius) {
          continue;
        }

        const int sample_y = ClampCoord(y + dy, height);
        const int sample_x = ClampCoord(x + dx, width);
        if (FC(pattern, sample_y, sample_x) != target_color) {
          continue;
        }

        const float weight = 1.0f / static_cast<float>(abs(dx) + abs(dy));
        weighted_sum += SafeRawRead(raw, width, height, sample_y, sample_x) * weight;
        weight_sum += weight;
      }
    }

    if (weight_sum > 0.0f) {
      break;
    }
  }

  if (weight_sum > 0.0f) {
    return weighted_sum / weight_sum;
  }
  return raw.ptr(y)[x];
}

__global__ void RCD_InitAndVHKernel(const cv::cuda::PtrStep<float> raw, cv::cuda::PtrStep<float> r,
                                    cv::cuda::PtrStep<float> g, cv::cuda::PtrStep<float> b,
                                    cv::cuda::PtrStep<float> vh_dir, cv::cuda::PtrStep<float> pq_dir,
                                    const int width,
                                    const int height, BayerPattern2x2 pattern) {
  __shared__ float raw_tile[kTileHeight][kTileWidth];
  LoadRawTileFull(raw, width, height, raw_tile);
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int   tx    = threadIdx.x + kRcdRadius;
  const int   ty    = threadIdx.y + kRcdRadius;
  const float val   = raw_tile[ty][tx];
  const int   color = FC(pattern, y, x);

  r.ptr(y)[x]       = (color == 0) ? val : 0.0f;
  g.ptr(y)[x]       = (color == 1) ? val : 0.0f;
  b.ptr(y)[x]       = (color == 2) ? val : 0.0f;
  pq_dir.ptr(y)[x]  = 0.0f;

  float vh          = 0.0f;
  if (y >= kRcdRadius && y < height - kRcdRadius && x >= kRcdRadius && x < width - kRcdRadius) {
    const float c   = val;

    const float vm1 = raw_tile[ty - 1][tx];
    const float vp1 = raw_tile[ty + 1][tx];
    const float vm2 = raw_tile[ty - 2][tx];
    const float vp2 = raw_tile[ty + 2][tx];
    const float vm3 = raw_tile[ty - 3][tx];
    const float vp3 = raw_tile[ty + 3][tx];
    const float vm4 = raw_tile[ty - 4][tx];
    const float vp4 = raw_tile[ty + 4][tx];

    const float hm1 = raw_tile[ty][tx - 1];
    const float hp1 = raw_tile[ty][tx + 1];
    const float hm2 = raw_tile[ty][tx - 2];
    const float hp2 = raw_tile[ty][tx + 2];
    const float hm3 = raw_tile[ty][tx - 3];
    const float hp3 = raw_tile[ty][tx + 3];
    const float hm4 = raw_tile[ty][tx - 4];
    const float hp4 = raw_tile[ty][tx + 4];

    const float V_stat = fmaxf(
        -18.f * c * vm1 - 18.f * c * vp1 - 36.f * c * vm2 - 36.f * c * vp2 + 18.f * c * vm3 +
            18.f * c * vp3 - 2.f * c * vm4 - 2.f * c * vp4 + 38.f * c * c - 70.f * vm1 * vp1 -
            12.f * vm1 * vm2 + 24.f * vm1 * vp2 - 38.f * vm1 * vm3 + 16.f * vm1 * vp3 +
            12.f * vm1 * vm4 - 6.f * vm1 * vp4 + 46.f * vm1 * vm1 + 24.f * vp1 * vm2 -
            12.f * vp1 * vp2 + 16.f * vp1 * vm3 - 38.f * vp1 * vp3 - 6.f * vp1 * vm4 +
            12.f * vp1 * vp4 + 46.f * vp1 * vp1 + 14.f * vm2 * vp2 - 12.f * vm2 * vp3 -
            2.f * vm2 * vm4 + 2.f * vm2 * vp4 + 11.f * vm2 * vm2 - 12.f * vp2 * vm3 +
            2.f * vp2 * vm4 - 2.f * vp2 * vp4 + 11.f * vp2 * vp2 + 2.f * vm3 * vp3 -
            6.f * vm3 * vm4 + 10.f * vm3 * vm3 - 6.f * vp3 * vp4 + 10.f * vp3 * vp3 +
            1.f * vm4 * vm4 + 1.f * vp4 * vp4,
        kEpsSq);

    const float H_stat = fmaxf(
        -18.f * c * hm1 - 18.f * c * hp1 - 36.f * c * hm2 - 36.f * c * hp2 + 18.f * c * hm3 +
            18.f * c * hp3 - 2.f * c * hm4 - 2.f * c * hp4 + 38.f * c * c - 70.f * hm1 * hp1 -
            12.f * hm1 * hm2 + 24.f * hm1 * hp2 - 38.f * hm1 * hm3 + 16.f * hm1 * hp3 +
            12.f * hm1 * hm4 - 6.f * hm1 * hp4 + 46.f * hm1 * hm1 + 24.f * hp1 * hm2 -
            12.f * hp1 * hp2 + 16.f * hp1 * hm3 - 38.f * hp1 * hp3 - 6.f * hp1 * hm4 +
            12.f * hp1 * hp4 + 46.f * hp1 * hp1 + 14.f * hm2 * hp2 - 12.f * hm2 * hp3 -
            2.f * hm2 * hm4 + 2.f * hm2 * hp4 + 11.f * hm2 * hm2 - 12.f * hp2 * hm3 +
            2.f * hp2 * hm4 - 2.f * hp2 * hp4 + 11.f * hp2 * hp2 + 2.f * hm3 * hp3 -
            6.f * hm3 * hm4 + 10.f * hm3 * hm3 - 6.f * hp3 * hp4 + 10.f * hp3 * hp3 +
            1.f * hm4 * hm4 + 1.f * hp4 * hp4,
        kEpsSq);

    vh = V_stat / (V_stat + H_stat);
  }

  vh_dir.ptr(y)[x] = vh;
}

__global__ void RCD_GreenAtRBKernel(const cv::cuda::PtrStep<float> raw,
                                    const cv::cuda::PtrStep<float> vh_dir, cv::cuda::PtrStep<float> g,
                                    const int width, const int height, BayerPattern2x2 pattern) {
  __shared__ float raw_tile[kTileHeight][kTileWidth];
  LoadRawTileFull(raw, width, height, raw_tile);
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < kRcdRadius || y >= height - kRcdRadius || x < kRcdRadius || x >= width - kRcdRadius) return;

  const int color = FC(pattern, y, x);
  if (color == 1) return;

  const float VH_central = vh_dir.ptr(y)[x];
  const float VH_neigh   = 0.25f * (vh_dir.ptr(y - 1)[x - 1] + vh_dir.ptr(y - 1)[x + 1] +
                                  vh_dir.ptr(y + 1)[x - 1] + vh_dir.ptr(y + 1)[x + 1]);
  const float VH_disc =
      (fabsf(0.5f - VH_central) < fabsf(0.5f - VH_neigh)) ? VH_neigh : VH_central;

  const int tx = threadIdx.x + kRcdRadius;
  const int ty = threadIdx.y + kRcdRadius;

  const float c   = raw_tile[ty][tx];

  const float vm1 = raw_tile[ty - 1][tx];
  const float vp1 = raw_tile[ty + 1][tx];
  const float vm2 = raw_tile[ty - 2][tx];
  const float vp2 = raw_tile[ty + 2][tx];
  const float vm3 = raw_tile[ty - 3][tx];
  const float vp3 = raw_tile[ty + 3][tx];
  const float vm4 = raw_tile[ty - 4][tx];
  const float vp4 = raw_tile[ty + 4][tx];

  const float hm1 = raw_tile[ty][tx - 1];
  const float hp1 = raw_tile[ty][tx + 1];
  const float hm2 = raw_tile[ty][tx - 2];
  const float hp2 = raw_tile[ty][tx + 2];
  const float hm3 = raw_tile[ty][tx - 3];
  const float hp3 = raw_tile[ty][tx + 3];
  const float hm4 = raw_tile[ty][tx - 4];
  const float hp4 = raw_tile[ty][tx + 4];

  const float lpf_c  = LowPassAtTile(raw_tile, ty, tx);
  const float lpf_n2 = LowPassAtTile(raw_tile, ty - 2, tx);
  const float lpf_s2 = LowPassAtTile(raw_tile, ty + 2, tx);
  const float lpf_w2 = LowPassAtTile(raw_tile, ty, tx - 2);
  const float lpf_e2 = LowPassAtTile(raw_tile, ty, tx + 2);

  const float N_grad = kEps + fabsf(vm1 - vp1) + fabsf(c - vm2) + fabsf(vm1 - vm3) + fabsf(vm2 - vm4);
  const float S_grad = kEps + fabsf(vp1 - vm1) + fabsf(c - vp2) + fabsf(vp1 - vp3) + fabsf(vp2 - vp4);
  const float W_grad = kEps + fabsf(hm1 - hp1) + fabsf(c - hm2) + fabsf(hm1 - hm3) + fabsf(hm2 - hm4);
  const float E_grad = kEps + fabsf(hp1 - hm1) + fabsf(c - hp2) + fabsf(hp1 - hp3) + fabsf(hp2 - hp4);

  const float N_est  = vm1 * (1.f + (lpf_c - lpf_n2) / (kEps + lpf_c + lpf_n2));
  const float S_est  = vp1 * (1.f + (lpf_c - lpf_s2) / (kEps + lpf_c + lpf_s2));
  const float W_est  = hm1 * (1.f + (lpf_c - lpf_w2) / (kEps + lpf_c + lpf_w2));
  const float E_est  = hp1 * (1.f + (lpf_c - lpf_e2) / (kEps + lpf_c + lpf_e2));

  const float V_est  = (S_grad * N_est + N_grad * S_est) / (N_grad + S_grad);
  const float H_est  = (W_grad * E_est + E_grad * W_est) / (E_grad + W_grad);

  g.ptr(y)[x]        = fmaxf(VH_disc * H_est + (1.f - VH_disc) * V_est, 0.f);
}

__global__ void RCD_PQDirKernel(const cv::cuda::PtrStep<float> raw, cv::cuda::PtrStep<float> pq_dir,
                                const int width, const int height, BayerPattern2x2 pattern) {
  __shared__ float raw_tile[kTileHeight][kTileWidth];
  LoadRawTileFull(raw, width, height, raw_tile);
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float pq = 0.0f;
  if (y >= kRcdRadius && y < height - kRcdRadius && x >= kRcdRadius && x < width - kRcdRadius) {
    const int color = FC(pattern, y, x);
    if (color != 1) {
      const int tx = threadIdx.x + kRcdRadius;
      const int ty = threadIdx.y + kRcdRadius;

      const float c   = raw_tile[ty][tx];

      const float nw1 = raw_tile[ty - 1][tx - 1];
      const float se1 = raw_tile[ty + 1][tx + 1];
      const float nw2 = raw_tile[ty - 2][tx - 2];
      const float se2 = raw_tile[ty + 2][tx + 2];
      const float nw3 = raw_tile[ty - 3][tx - 3];
      const float se3 = raw_tile[ty + 3][tx + 3];
      const float nw4 = raw_tile[ty - 4][tx - 4];
      const float se4 = raw_tile[ty + 4][tx + 4];

      const float sw1 = raw_tile[ty + 1][tx - 1];
      const float ne1 = raw_tile[ty - 1][tx + 1];
      const float sw2 = raw_tile[ty + 2][tx - 2];
      const float ne2 = raw_tile[ty - 2][tx + 2];
      const float sw3 = raw_tile[ty + 3][tx - 3];
      const float ne3 = raw_tile[ty - 3][tx + 3];
      const float sw4 = raw_tile[ty + 4][tx - 4];
      const float ne4 = raw_tile[ty - 4][tx + 4];

      const float P_stat = fmaxf(
          -18.f * c * nw1 - 18.f * c * se1 - 36.f * c * nw2 - 36.f * c * se2 + 18.f * c * nw3 +
              18.f * c * se3 - 2.f * c * nw4 - 2.f * c * se4 + 38.f * c * c -
              70.f * nw1 * se1 - 12.f * nw1 * nw2 + 24.f * nw1 * se2 - 38.f * nw1 * nw3 +
              16.f * nw1 * se3 + 12.f * nw1 * nw4 - 6.f * nw1 * se4 + 46.f * nw1 * nw1 +
              24.f * se1 * nw2 - 12.f * se1 * se2 + 16.f * se1 * nw3 - 38.f * se1 * se3 -
              6.f * se1 * nw4 + 12.f * se1 * se4 + 46.f * se1 * se1 + 14.f * nw2 * se2 -
              12.f * nw2 * se3 - 2.f * nw2 * nw4 + 2.f * nw2 * se4 + 11.f * nw2 * nw2 -
              12.f * se2 * nw3 + 2.f * se2 * nw4 - 2.f * se2 * se4 + 11.f * se2 * se2 +
              2.f * nw3 * se3 - 6.f * nw3 * nw4 + 10.f * nw3 * nw3 - 6.f * se3 * se4 +
              10.f * se3 * se3 + 1.f * nw4 * nw4 + 1.f * se4 * se4,
          kEpsSq);

      const float Q_stat = fmaxf(
          -18.f * c * sw1 - 18.f * c * ne1 - 36.f * c * sw2 - 36.f * c * ne2 + 18.f * c * sw3 +
              18.f * c * ne3 - 2.f * c * sw4 - 2.f * c * ne4 + 38.f * c * c -
              70.f * sw1 * ne1 - 12.f * sw1 * sw2 + 24.f * sw1 * ne2 - 38.f * sw1 * sw3 +
              16.f * sw1 * ne3 + 12.f * sw1 * sw4 - 6.f * sw1 * ne4 + 46.f * sw1 * sw1 +
              24.f * ne1 * sw2 - 12.f * ne1 * ne2 + 16.f * ne1 * sw3 - 38.f * ne1 * ne3 -
              6.f * ne1 * sw4 + 12.f * ne1 * ne4 + 46.f * ne1 * ne1 + 14.f * sw2 * ne2 -
              12.f * sw2 * ne3 - 2.f * sw2 * sw4 + 2.f * sw2 * ne4 + 11.f * sw2 * sw2 -
              12.f * ne2 * sw3 + 2.f * ne2 * sw4 - 2.f * ne2 * ne4 + 11.f * ne2 * ne2 +
              2.f * sw3 * ne3 - 6.f * sw3 * sw4 + 10.f * sw3 * sw3 - 6.f * ne3 * ne4 +
              10.f * ne3 * ne3 + 1.f * sw4 * sw4 + 1.f * ne4 * ne4,
          kEpsSq);

      pq = P_stat / (P_stat + Q_stat);
    }
  }

  pq_dir.ptr(y)[x] = pq;
}

__global__ void RCD_RBAtRBKernel(const cv::cuda::PtrStep<float> pq_dir, const cv::cuda::PtrStep<float> g,
                                 cv::cuda::PtrStep<float> r, cv::cuda::PtrStep<float> b,
                                 const int width, const int height, BayerPattern2x2 pattern) {
  __shared__ float pq_tile[kThreadsY + 2][kThreadsX + 2];
  __shared__ float g_tile[kThreadsY + 4][kThreadsX + 4];
  __shared__ float r_tile[kThreadsY + 7][kThreadsX + 7];
  __shared__ float b_tile[kThreadsY + 7][kThreadsX + 7];

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 1;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 1;
    for (int ty = threadIdx.y; ty < (kThreadsY + 2); ty += blockDim.y) {
      const int    gy      = ClampCoord(tile_y0 + ty, height);
      const float* pq_row  = pq_dir.ptr(gy);
      float*       tile_row = pq_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 2); tx += blockDim.x) {
        tile_row[tx] = pq_row[ClampCoord(tile_x0 + tx, width)];
      }
    }
  }

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 2;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 2;
    for (int ty = threadIdx.y; ty < (kThreadsY + 4); ty += blockDim.y) {
      const int    gy      = ClampCoord(tile_y0 + ty, height);
      const float* g_row   = g.ptr(gy);
      float*       tile_row = g_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 4); tx += blockDim.x) {
        tile_row[tx] = g_row[ClampCoord(tile_x0 + tx, width)];
      }
    }
  }

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 3;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 3;
    for (int ty = threadIdx.y; ty < (kThreadsY + 7); ty += blockDim.y) {
      const int    gy       = ClampCoord(tile_y0 + ty, height);
      const float* r_row    = r.ptr(gy);
      const float* b_row    = b.ptr(gy);
      float*       r_trow   = r_tile[ty];
      float*       b_trow   = b_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 7); tx += blockDim.x) {
        const int gx = ClampCoord(tile_x0 + tx, width);
        r_trow[tx] = r_row[gx];
        b_trow[tx] = b_row[gx];
      }
    }
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < kRcdRadius || y >= height - kRcdRadius || x < kRcdRadius || x >= width - kRcdRadius) return;

  const int color = FC(pattern, y, x);
  if (color == 1) return;

  const int   c        = 2 - color;  // missing channel at RB position (R->B, B->R)

  const int ptx = threadIdx.x + 1;
  const int pty = threadIdx.y + 1;
  const int gtx = threadIdx.x + 2;
  const int gty = threadIdx.y + 2;
  const int ctx = threadIdx.x + 3;
  const int cty = threadIdx.y + 3;

  const float PQ_c     = pq_tile[pty][ptx];
  const float PQ_n     = 0.25f * (pq_tile[pty - 1][ptx - 1] + pq_tile[pty - 1][ptx + 1] +
                                 pq_tile[pty + 1][ptx - 1] + pq_tile[pty + 1][ptx + 1]);
  const float PQ_disc  = (fabsf(0.5f - PQ_c) < fabsf(0.5f - PQ_n)) ? PQ_n : PQ_c;

  const float g_c      = g_tile[gty][gtx];

  const float ch_nw1   = (c == 0) ? r_tile[cty - 1][ctx - 1] : b_tile[cty - 1][ctx - 1];
  const float ch_ne1   = (c == 0) ? r_tile[cty - 1][ctx + 1] : b_tile[cty - 1][ctx + 1];
  const float ch_sw1   = (c == 0) ? r_tile[cty + 1][ctx - 1] : b_tile[cty + 1][ctx - 1];
  const float ch_se1   = (c == 0) ? r_tile[cty + 1][ctx + 1] : b_tile[cty + 1][ctx + 1];

  const float ch_nw3   = (c == 0) ? r_tile[cty - 3][ctx - 3] : b_tile[cty - 3][ctx - 3];
  const float ch_ne3   = (c == 0) ? r_tile[cty - 3][ctx + 3] : b_tile[cty - 3][ctx + 3];
  const float ch_sw3   = (c == 0) ? r_tile[cty + 3][ctx - 3] : b_tile[cty + 3][ctx - 3];
  const float ch_se3   = (c == 0) ? r_tile[cty + 3][ctx + 3] : b_tile[cty + 3][ctx + 3];

  const float g_nw2    = g_tile[gty - 2][gtx - 2];
  const float g_ne2    = g_tile[gty - 2][gtx + 2];
  const float g_sw2    = g_tile[gty + 2][gtx - 2];
  const float g_se2    = g_tile[gty + 2][gtx + 2];

  const float NW_grad  = kEps + fabsf(ch_nw1 - ch_se1) + fabsf(ch_nw1 - ch_nw3) + fabsf(g_c - g_nw2);
  const float NE_grad  = kEps + fabsf(ch_ne1 - ch_sw1) + fabsf(ch_ne1 - ch_ne3) + fabsf(g_c - g_ne2);
  const float SW_grad  = kEps + fabsf(ch_sw1 - ch_ne1) + fabsf(ch_sw1 - ch_sw3) + fabsf(g_c - g_sw2);
  const float SE_grad  = kEps + fabsf(ch_se1 - ch_nw1) + fabsf(ch_se1 - ch_se3) + fabsf(g_c - g_se2);

  const float g_nw1    = g_tile[gty - 1][gtx - 1];
  const float g_ne1    = g_tile[gty - 1][gtx + 1];
  const float g_sw1    = g_tile[gty + 1][gtx - 1];
  const float g_se1    = g_tile[gty + 1][gtx + 1];

  const float NW_est   = ch_nw1 - g_nw1;
  const float NE_est   = ch_ne1 - g_ne1;
  const float SW_est   = ch_sw1 - g_sw1;
  const float SE_est   = ch_se1 - g_se1;

  const float P_est    = (NW_grad * SE_est + SE_grad * NW_est) / (NW_grad + SE_grad);
  const float Q_est    = (NE_grad * SW_est + SW_grad * NE_est) / (NE_grad + SW_grad);

  const float out_val  = fmaxf(0.f, g_c + (1.f - PQ_disc) * P_est + PQ_disc * Q_est);

  if (c == 0) {
    r.ptr(y)[x] = out_val;
  } else {
    b.ptr(y)[x] = out_val;
  }
}

__global__ void RCD_RBAtGKernel(const cv::cuda::PtrStep<float> vh_dir, const cv::cuda::PtrStep<float> g,
                                cv::cuda::PtrStep<float> r, cv::cuda::PtrStep<float> b,
                                const int width, const int height, BayerPattern2x2 pattern) {
  __shared__ float vh_tile[kThreadsY + 2][kThreadsX + 2];
  __shared__ float g_tile[kThreadsY + 4][kThreadsX + 4];
  __shared__ float r_tile[kThreadsY + 7][kThreadsX + 7];
  __shared__ float b_tile[kThreadsY + 7][kThreadsX + 7];

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 1;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 1;
    for (int ty = threadIdx.y; ty < (kThreadsY + 2); ty += blockDim.y) {
      const int    gy      = ClampCoord(tile_y0 + ty, height);
      const float* vh_row  = vh_dir.ptr(gy);
      float*       tile_row = vh_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 2); tx += blockDim.x) {
        tile_row[tx] = vh_row[ClampCoord(tile_x0 + tx, width)];
      }
    }
  }

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 2;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 2;
    for (int ty = threadIdx.y; ty < (kThreadsY + 4); ty += blockDim.y) {
      const int    gy      = ClampCoord(tile_y0 + ty, height);
      const float* g_row   = g.ptr(gy);
      float*       tile_row = g_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 4); tx += blockDim.x) {
        tile_row[tx] = g_row[ClampCoord(tile_x0 + tx, width)];
      }
    }
  }

  {
    const int tile_x0 = static_cast<int>(blockIdx.x) * blockDim.x - 3;
    const int tile_y0 = static_cast<int>(blockIdx.y) * blockDim.y - 3;
    for (int ty = threadIdx.y; ty < (kThreadsY + 7); ty += blockDim.y) {
      const int    gy      = ClampCoord(tile_y0 + ty, height);
      const float* r_row   = r.ptr(gy);
      const float* b_row   = b.ptr(gy);
      float*       r_trow  = r_tile[ty];
      float*       b_trow  = b_tile[ty];
      for (int tx = threadIdx.x; tx < (kThreadsX + 7); tx += blockDim.x) {
        const int gx = ClampCoord(tile_x0 + tx, width);
        r_trow[tx] = r_row[gx];
        b_trow[tx] = b_row[gx];
      }
    }
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < kRcdRadius || y >= height - kRcdRadius || x < kRcdRadius || x >= width - kRcdRadius) return;

  const int color = FC(pattern, y, x);
  if (color != 1) return;

  const int vtx = threadIdx.x + 1;
  const int vty = threadIdx.y + 1;
  const int gtx = threadIdx.x + 2;
  const int gty = threadIdx.y + 2;
  const int ctx = threadIdx.x + 3;
  const int cty = threadIdx.y + 3;

  const float VH_central = vh_tile[vty][vtx];
  const float VH_neigh   = 0.25f * (vh_tile[vty - 1][vtx - 1] + vh_tile[vty - 1][vtx + 1] +
                                   vh_tile[vty + 1][vtx - 1] + vh_tile[vty + 1][vtx + 1]);
  const float VH_disc =
      (fabsf(0.5f - VH_central) < fabsf(0.5f - VH_neigh)) ? VH_neigh : VH_central;

  const float g_c = g_tile[gty][gtx];

  const float g_m2 = g_tile[gty - 2][gtx];
  const float g_p2 = g_tile[gty + 2][gtx];
  const float g_l2 = g_tile[gty][gtx - 2];
  const float g_r2 = g_tile[gty][gtx + 2];

  // --- R @ G ---
  {
    const float ch_m1  = r_tile[cty - 1][ctx];
    const float ch_p1  = r_tile[cty + 1][ctx];
    const float ch_m3  = r_tile[cty - 3][ctx];
    const float ch_p3  = r_tile[cty + 3][ctx];
    const float ch_l1  = r_tile[cty][ctx - 1];
    const float ch_r1  = r_tile[cty][ctx + 1];
    const float ch_l3  = r_tile[cty][ctx - 3];
    const float ch_r3  = r_tile[cty][ctx + 3];

    const float N_grad = kEps + fabsf(g_c - g_m2) + fabsf(ch_m1 - ch_p1) + fabsf(ch_m1 - ch_m3);
    const float S_grad = kEps + fabsf(g_c - g_p2) + fabsf(ch_p1 - ch_m1) + fabsf(ch_p1 - ch_p3);
    const float W_grad = kEps + fabsf(g_c - g_l2) + fabsf(ch_l1 - ch_r1) + fabsf(ch_l1 - ch_l3);
    const float E_grad = kEps + fabsf(g_c - g_r2) + fabsf(ch_r1 - ch_l1) + fabsf(ch_r1 - ch_r3);

    const float N_est  = ch_m1 - g_tile[gty - 1][gtx];
    const float S_est  = ch_p1 - g_tile[gty + 1][gtx];
    const float W_est  = ch_l1 - g_tile[gty][gtx - 1];
    const float E_est  = ch_r1 - g_tile[gty][gtx + 1];

    const float V_est  = (N_grad * S_est + S_grad * N_est) / (N_grad + S_grad);
    const float H_est  = (E_grad * W_est + W_grad * E_est) / (E_grad + W_grad);

    r.ptr(y)[x]        = fmaxf(0.f, g_c + (1.f - VH_disc) * V_est + VH_disc * H_est);
  }

  // --- B @ G ---
  {
    const float ch_m1  = b_tile[cty - 1][ctx];
    const float ch_p1  = b_tile[cty + 1][ctx];
    const float ch_m3  = b_tile[cty - 3][ctx];
    const float ch_p3  = b_tile[cty + 3][ctx];
    const float ch_l1  = b_tile[cty][ctx - 1];
    const float ch_r1  = b_tile[cty][ctx + 1];
    const float ch_l3  = b_tile[cty][ctx - 3];
    const float ch_r3  = b_tile[cty][ctx + 3];

    const float N_grad = kEps + fabsf(g_c - g_m2) + fabsf(ch_m1 - ch_p1) + fabsf(ch_m1 - ch_m3);
    const float S_grad = kEps + fabsf(g_c - g_p2) + fabsf(ch_p1 - ch_m1) + fabsf(ch_p1 - ch_p3);
    const float W_grad = kEps + fabsf(g_c - g_l2) + fabsf(ch_l1 - ch_r1) + fabsf(ch_l1 - ch_l3);
    const float E_grad = kEps + fabsf(g_c - g_r2) + fabsf(ch_r1 - ch_l1) + fabsf(ch_r1 - ch_r3);

    const float N_est  = ch_m1 - g_tile[gty - 1][gtx];
    const float S_est  = ch_p1 - g_tile[gty + 1][gtx];
    const float W_est  = ch_l1 - g_tile[gty][gtx - 1];
    const float E_est  = ch_r1 - g_tile[gty][gtx + 1];

    const float V_est  = (N_grad * S_est + S_grad * N_est) / (N_grad + S_grad);
    const float H_est  = (E_grad * W_est + W_grad * E_est) / (E_grad + W_grad);

    b.ptr(y)[x]        = fmaxf(0.f, g_c + (1.f - VH_disc) * V_est + VH_disc * H_est);
  }
}

__global__ void RCD_FillEdgeKernel(const cv::cuda::PtrStep<float> raw, cv::cuda::PtrStep<float> r,
                                   cv::cuda::PtrStep<float> g, cv::cuda::PtrStep<float> b,
                                   const int width, const int height, BayerPattern2x2 pattern) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  if (y >= kEdgeFallbackRadius && y < height - kEdgeFallbackRadius && x >= kEdgeFallbackRadius &&
      x < width - kEdgeFallbackRadius) {
    return;
  }

  // RCD needs a 4-pixel neighborhood. Fill the border band with a simple CFA-aware fallback
  // instead of cropping away valid output pixels.
  r.ptr(y)[x] = EstimateEdgeChannel(raw, width, height, pattern, y, x, 0);
  g.ptr(y)[x] = EstimateEdgeChannel(raw, width, height, pattern, y, x, 1);
  b.ptr(y)[x] = EstimateEdgeChannel(raw, width, height, pattern, y, x, 2);
}

}  // namespace

void Bayer2x2ToRGB_RCD(cv::cuda::GpuMat& image, const BayerPattern2x2& pattern,
                       RcdWorkspace* workspace, cv::cuda::Stream* stream) {
  CV_Assert(image.type() == CV_32FC1);

  const int width  = image.cols;
  const int height = image.rows;

  if (width <= 0 || height <= 0) return;

  RcdWorkspace local_workspace;
  RcdWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;
  active_workspace.Reserve(cv::Size(width, height));

  cv::cuda::GpuMat& r      = active_workspace.r;
  cv::cuda::GpuMat& g      = active_workspace.g;
  cv::cuda::GpuMat& b      = active_workspace.b;
  cv::cuda::GpuMat& vh_dir = active_workspace.vh_dir;
  cv::cuda::GpuMat& pq_dir = active_workspace.pq_dir;

  cv::cuda::Stream local_stream;
  cv::cuda::Stream& active_stream = stream == nullptr ? local_stream : *stream;
  cudaStream_t     cuda_stream = cv::cuda::StreamAccessor::getStream(active_stream);

  const dim3       threads(kThreadsX, kThreadsY);
  const dim3       blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  RCD_InitAndVHKernel<<<blocks, threads, 0, cuda_stream>>>(image, r, g, b, vh_dir, pq_dir, width,
                                                            height, pattern);
  CUDA_CHECK(cudaGetLastError());

  RCD_GreenAtRBKernel<<<blocks, threads, 0, cuda_stream>>>(image, vh_dir, g, width, height,
                                                           pattern);
  CUDA_CHECK(cudaGetLastError());

  RCD_PQDirKernel<<<blocks, threads, 0, cuda_stream>>>(image, pq_dir, width, height, pattern);
  CUDA_CHECK(cudaGetLastError());

  RCD_RBAtRBKernel<<<blocks, threads, 0, cuda_stream>>>(pq_dir, g, r, b, width, height, pattern);
  CUDA_CHECK(cudaGetLastError());

  RCD_RBAtGKernel<<<blocks, threads, 0, cuda_stream>>>(vh_dir, g, r, b, width, height, pattern);
  CUDA_CHECK(cudaGetLastError());

  RCD_FillEdgeKernel<<<blocks, threads, 0, cuda_stream>>>(image, r, g, b, width, height, pattern);
  CUDA_CHECK(cudaGetLastError());

  MergeRGB(r, g, b, image, &active_stream);
  if (stream == nullptr) {
    active_stream.waitForCompletion();
  }
}
}
}
