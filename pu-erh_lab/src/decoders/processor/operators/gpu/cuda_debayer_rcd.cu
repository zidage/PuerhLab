//  Copyright 2026 Yurun Zi
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

/**
* RATIO CORRECTED DEMOSAICING
* Luis Sanz Rodríguez (luis.sanz.rodriguez(at)gmail(dot)com)
*
* Release 2.3 @ 171125
*/

#include "decoders/processor/operators/gpu/cuda_debayer_rcd.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaarithm.hpp>

#include "decoders/processor/operators/gpu/cuda_raw_proc_utils.hpp"

namespace puerhlab {
namespace CUDA {

namespace {

constexpr float kEps   = 1e-5f;
constexpr float kEpsSq = 1e-10f;

__device__ __forceinline__ int FC(const int y, const int x) {
  // Matches CPU debayer_rcd.cpp: static int fc[2][2] = {{0, 1}, {1, 2}};
  return (y & 1) ? ((x & 1) ? 2 : 1) : ((x & 1) ? 1 : 0);
}

__device__ __forceinline__ float LowPassAt(const cv::cuda::PtrStep<float> raw, const int y,
                                           const int x) {
  const float* row_m1 = raw.ptr(y - 1);
  const float* row_0  = raw.ptr(y);
  const float* row_p1 = raw.ptr(y + 1);

  const float  c      = row_0[x];
  const float  n      = row_m1[x];
  const float  s      = row_p1[x];
  const float  w      = row_0[x - 1];
  const float  e      = row_0[x + 1];

  const float  nw     = row_m1[x - 1];
  const float  ne     = row_m1[x + 1];
  const float  sw     = row_p1[x - 1];
  const float  se     = row_p1[x + 1];

  return 0.25f * c + 0.125f * (n + s + w + e) + 0.0625f * (nw + ne + sw + se);
}

__global__ void RCD_InitAndVHKernel(const cv::cuda::PtrStep<float> raw, cv::cuda::PtrStep<float> r,
                                   cv::cuda::PtrStep<float> g, cv::cuda::PtrStep<float> b,
                                   cv::cuda::PtrStep<float> vh_dir, const int width,
                                   const int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const float val   = raw.ptr(y)[x];
  const int   color = FC(y, x);

  r.ptr(y)[x]       = (color == 0) ? val : 0.0f;
  g.ptr(y)[x]       = (color == 1) ? val : 0.0f;
  b.ptr(y)[x]       = (color == 2) ? val : 0.0f;

  float vh          = 0.0f;
  if (y >= 4 && y < height - 4 && x >= 4 && x < width - 4) {
    const float c   = val;

    const float vm1 = raw.ptr(y - 1)[x];
    const float vp1 = raw.ptr(y + 1)[x];
    const float vm2 = raw.ptr(y - 2)[x];
    const float vp2 = raw.ptr(y + 2)[x];
    const float vm3 = raw.ptr(y - 3)[x];
    const float vp3 = raw.ptr(y + 3)[x];
    const float vm4 = raw.ptr(y - 4)[x];
    const float vp4 = raw.ptr(y + 4)[x];

    const float hm1 = raw.ptr(y)[x - 1];
    const float hp1 = raw.ptr(y)[x + 1];
    const float hm2 = raw.ptr(y)[x - 2];
    const float hp2 = raw.ptr(y)[x + 2];
    const float hm3 = raw.ptr(y)[x - 3];
    const float hp3 = raw.ptr(y)[x + 3];
    const float hm4 = raw.ptr(y)[x - 4];
    const float hp4 = raw.ptr(y)[x + 4];

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
                                   const int width, const int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < 4 || y >= height - 4 || x < 4 || x >= width - 4) return;

  const int color = FC(y, x);
  if (color == 1) return;

  const float VH_central = vh_dir.ptr(y)[x];
  const float VH_neigh   = 0.25f * (vh_dir.ptr(y - 1)[x - 1] + vh_dir.ptr(y - 1)[x + 1] +
                                  vh_dir.ptr(y + 1)[x - 1] + vh_dir.ptr(y + 1)[x + 1]);
  const float VH_disc =
      (fabsf(0.5f - VH_central) < fabsf(0.5f - VH_neigh)) ? VH_neigh : VH_central;

  const float c   = raw.ptr(y)[x];

  const float vm1 = raw.ptr(y - 1)[x];
  const float vp1 = raw.ptr(y + 1)[x];
  const float vm2 = raw.ptr(y - 2)[x];
  const float vp2 = raw.ptr(y + 2)[x];
  const float vm3 = raw.ptr(y - 3)[x];
  const float vp3 = raw.ptr(y + 3)[x];
  const float vm4 = raw.ptr(y - 4)[x];
  const float vp4 = raw.ptr(y + 4)[x];

  const float hm1 = raw.ptr(y)[x - 1];
  const float hp1 = raw.ptr(y)[x + 1];
  const float hm2 = raw.ptr(y)[x - 2];
  const float hp2 = raw.ptr(y)[x + 2];
  const float hm3 = raw.ptr(y)[x - 3];
  const float hp3 = raw.ptr(y)[x + 3];
  const float hm4 = raw.ptr(y)[x - 4];
  const float hp4 = raw.ptr(y)[x + 4];

  const float lpf_c  = LowPassAt(raw, y, x);
  const float lpf_n2 = LowPassAt(raw, y - 2, x);
  const float lpf_s2 = LowPassAt(raw, y + 2, x);
  const float lpf_w2 = LowPassAt(raw, y, x - 2);
  const float lpf_e2 = LowPassAt(raw, y, x + 2);

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
                               const int width, const int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float pq = 0.0f;
  if (y >= 4 && y < height - 4 && x >= 4 && x < width - 4) {
    const int color = FC(y, x);
    if (color != 1) {
      const float c   = raw.ptr(y)[x];

      const float nw1 = raw.ptr(y - 1)[x - 1];
      const float se1 = raw.ptr(y + 1)[x + 1];
      const float nw2 = raw.ptr(y - 2)[x - 2];
      const float se2 = raw.ptr(y + 2)[x + 2];
      const float nw3 = raw.ptr(y - 3)[x - 3];
      const float se3 = raw.ptr(y + 3)[x + 3];
      const float nw4 = raw.ptr(y - 4)[x - 4];
      const float se4 = raw.ptr(y + 4)[x + 4];

      const float sw1 = raw.ptr(y + 1)[x - 1];
      const float ne1 = raw.ptr(y - 1)[x + 1];
      const float sw2 = raw.ptr(y + 2)[x - 2];
      const float ne2 = raw.ptr(y - 2)[x + 2];
      const float sw3 = raw.ptr(y + 3)[x - 3];
      const float ne3 = raw.ptr(y - 3)[x + 3];
      const float sw4 = raw.ptr(y + 4)[x - 4];
      const float ne4 = raw.ptr(y - 4)[x + 4];

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
                                const int width, const int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < 4 || y >= height - 4 || x < 4 || x >= width - 4) return;

  const int color = FC(y, x);
  if (color == 1) return;

  const int   c        = 2 - color;  // missing channel at RB position (R->B, B->R)

  const float PQ_c     = pq_dir.ptr(y)[x];
  const float PQ_n     = 0.25f * (pq_dir.ptr(y - 1)[x - 1] + pq_dir.ptr(y - 1)[x + 1] +
                                pq_dir.ptr(y + 1)[x - 1] + pq_dir.ptr(y + 1)[x + 1]);
  const float PQ_disc  = (fabsf(0.5f - PQ_c) < fabsf(0.5f - PQ_n)) ? PQ_n : PQ_c;

  const float g_c      = g.ptr(y)[x];

  const float ch_nw1   = (c == 0) ? r.ptr(y - 1)[x - 1] : b.ptr(y - 1)[x - 1];
  const float ch_ne1   = (c == 0) ? r.ptr(y - 1)[x + 1] : b.ptr(y - 1)[x + 1];
  const float ch_sw1   = (c == 0) ? r.ptr(y + 1)[x - 1] : b.ptr(y + 1)[x - 1];
  const float ch_se1   = (c == 0) ? r.ptr(y + 1)[x + 1] : b.ptr(y + 1)[x + 1];

  const float ch_nw3   = (c == 0) ? r.ptr(y - 3)[x - 3] : b.ptr(y - 3)[x - 3];
  const float ch_ne3   = (c == 0) ? r.ptr(y - 3)[x + 3] : b.ptr(y - 3)[x + 3];
  const float ch_sw3   = (c == 0) ? r.ptr(y + 3)[x - 3] : b.ptr(y + 3)[x - 3];
  const float ch_se3   = (c == 0) ? r.ptr(y + 3)[x + 3] : b.ptr(y + 3)[x + 3];

  const float g_nw2    = g.ptr(y - 2)[x - 2];
  const float g_ne2    = g.ptr(y - 2)[x + 2];
  const float g_sw2    = g.ptr(y + 2)[x - 2];
  const float g_se2    = g.ptr(y + 2)[x + 2];

  const float NW_grad  = kEps + fabsf(ch_nw1 - ch_se1) + fabsf(ch_nw1 - ch_nw3) + fabsf(g_c - g_nw2);
  const float NE_grad  = kEps + fabsf(ch_ne1 - ch_sw1) + fabsf(ch_ne1 - ch_ne3) + fabsf(g_c - g_ne2);
  const float SW_grad  = kEps + fabsf(ch_sw1 - ch_ne1) + fabsf(ch_sw1 - ch_sw3) + fabsf(g_c - g_sw2);
  const float SE_grad  = kEps + fabsf(ch_se1 - ch_nw1) + fabsf(ch_se1 - ch_se3) + fabsf(g_c - g_se2);

  const float g_nw1    = g.ptr(y - 1)[x - 1];
  const float g_ne1    = g.ptr(y - 1)[x + 1];
  const float g_sw1    = g.ptr(y + 1)[x - 1];
  const float g_se1    = g.ptr(y + 1)[x + 1];

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
                               const int width, const int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  if (y < 4 || y >= height - 4 || x < 4 || x >= width - 4) return;

  const int color = FC(y, x);
  if (color != 1) return;

  const float VH_central = vh_dir.ptr(y)[x];
  const float VH_neigh   = 0.25f * (vh_dir.ptr(y - 1)[x - 1] + vh_dir.ptr(y - 1)[x + 1] +
                                  vh_dir.ptr(y + 1)[x - 1] + vh_dir.ptr(y + 1)[x + 1]);
  const float VH_disc =
      (fabsf(0.5f - VH_central) < fabsf(0.5f - VH_neigh)) ? VH_neigh : VH_central;

  const float g_c = g.ptr(y)[x];

  const float g_m2 = g.ptr(y - 2)[x];
  const float g_p2 = g.ptr(y + 2)[x];
  const float g_l2 = g.ptr(y)[x - 2];
  const float g_r2 = g.ptr(y)[x + 2];

  // --- R @ G ---
  {
    const float ch_m1  = r.ptr(y - 1)[x];
    const float ch_p1  = r.ptr(y + 1)[x];
    const float ch_m3  = r.ptr(y - 3)[x];
    const float ch_p3  = r.ptr(y + 3)[x];
    const float ch_l1  = r.ptr(y)[x - 1];
    const float ch_r1  = r.ptr(y)[x + 1];
    const float ch_l3  = r.ptr(y)[x - 3];
    const float ch_r3  = r.ptr(y)[x + 3];

    const float N_grad = kEps + fabsf(g_c - g_m2) + fabsf(ch_m1 - ch_p1) + fabsf(ch_m1 - ch_m3);
    const float S_grad = kEps + fabsf(g_c - g_p2) + fabsf(ch_p1 - ch_m1) + fabsf(ch_p1 - ch_p3);
    const float W_grad = kEps + fabsf(g_c - g_l2) + fabsf(ch_l1 - ch_r1) + fabsf(ch_l1 - ch_l3);
    const float E_grad = kEps + fabsf(g_c - g_r2) + fabsf(ch_r1 - ch_l1) + fabsf(ch_r1 - ch_r3);

    const float N_est  = ch_m1 - g.ptr(y - 1)[x];
    const float S_est  = ch_p1 - g.ptr(y + 1)[x];
    const float W_est  = ch_l1 - g.ptr(y)[x - 1];
    const float E_est  = ch_r1 - g.ptr(y)[x + 1];

    const float V_est  = (N_grad * S_est + S_grad * N_est) / (N_grad + S_grad);
    const float H_est  = (E_grad * W_est + W_grad * E_est) / (E_grad + W_grad);

    r.ptr(y)[x]        = fmaxf(0.f, g_c + (1.f - VH_disc) * V_est + VH_disc * H_est);
  }

  // --- B @ G ---
  {
    const float ch_m1  = b.ptr(y - 1)[x];
    const float ch_p1  = b.ptr(y + 1)[x];
    const float ch_m3  = b.ptr(y - 3)[x];
    const float ch_p3  = b.ptr(y + 3)[x];
    const float ch_l1  = b.ptr(y)[x - 1];
    const float ch_r1  = b.ptr(y)[x + 1];
    const float ch_l3  = b.ptr(y)[x - 3];
    const float ch_r3  = b.ptr(y)[x + 3];

    const float N_grad = kEps + fabsf(g_c - g_m2) + fabsf(ch_m1 - ch_p1) + fabsf(ch_m1 - ch_m3);
    const float S_grad = kEps + fabsf(g_c - g_p2) + fabsf(ch_p1 - ch_m1) + fabsf(ch_p1 - ch_p3);
    const float W_grad = kEps + fabsf(g_c - g_l2) + fabsf(ch_l1 - ch_r1) + fabsf(ch_l1 - ch_l3);
    const float E_grad = kEps + fabsf(g_c - g_r2) + fabsf(ch_r1 - ch_l1) + fabsf(ch_r1 - ch_r3);

    const float N_est  = ch_m1 - g.ptr(y - 1)[x];
    const float S_est  = ch_p1 - g.ptr(y + 1)[x];
    const float W_est  = ch_l1 - g.ptr(y)[x - 1];
    const float E_est  = ch_r1 - g.ptr(y)[x + 1];

    const float V_est  = (N_grad * S_est + S_grad * N_est) / (N_grad + S_grad);
    const float H_est  = (E_grad * W_est + W_grad * E_est) / (E_grad + W_grad);

    b.ptr(y)[x]        = fmaxf(0.f, g_c + (1.f - VH_disc) * V_est + VH_disc * H_est);
  }
}

}  // namespace

void BayerRGGB2RGB_RCD(cv::cuda::GpuMat& image) {
  CV_Assert(image.type() == CV_32FC1);

  const int width  = image.cols;
  const int height = image.rows;

  if (width <= 0 || height <= 0) return;

  cv::cuda::GpuMat r(height, width, CV_32FC1);
  cv::cuda::GpuMat g(height, width, CV_32FC1);
  cv::cuda::GpuMat b(height, width, CV_32FC1);
  cv::cuda::GpuMat vh_dir(height, width, CV_32FC1);
  cv::cuda::GpuMat pq_dir(height, width, CV_32FC1);

  cv::cuda::Stream stream;
  cudaStream_t     cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  const dim3       threads(32, 8);
  const dim3       blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  RCD_InitAndVHKernel<<<blocks, threads, 0, cuda_stream>>>(image, r, g, b, vh_dir, width, height);
  CUDA_CHECK(cudaGetLastError());

  RCD_GreenAtRBKernel<<<blocks, threads, 0, cuda_stream>>>(image, vh_dir, g, width, height);
  CUDA_CHECK(cudaGetLastError());

  RCD_PQDirKernel<<<blocks, threads, 0, cuda_stream>>>(image, pq_dir, width, height);
  CUDA_CHECK(cudaGetLastError());

  RCD_RBAtRBKernel<<<blocks, threads, 0, cuda_stream>>>(pq_dir, g, r, b, width, height);
  CUDA_CHECK(cudaGetLastError());

  RCD_RBAtGKernel<<<blocks, threads, 0, cuda_stream>>>(vh_dir, g, r, b, width, height);
  CUDA_CHECK(cudaGetLastError());

  std::vector<cv::cuda::GpuMat> channels = {r, g, b};
  cv::cuda::merge(channels, image, stream);
  stream.waitForCompletion();
}
}
}
