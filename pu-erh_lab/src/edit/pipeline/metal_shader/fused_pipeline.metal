//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "common.metal"
#include "GPU_kernels/basic.metal"
#include "GPU_kernels/color.metal"
#include "GPU_kernels/cst.metal"
#include "GPU_kernels/detail.metal"

kernel void metal_fused_pipeline_rgba32f(texture2d<float, access::read> src [[texture(0)]],
                                         texture2d<float, access::write> dst [[texture(1)]],
                                         constant MetalFusedParams& params [[buffer(0)]],
                                         device const float4* lmt_lut [[buffer(1)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
    return;
  }

  float4 px = src.read(gid);
  px        = GPU_TOWS_Kernel(px, params);
  px        = GPU_ExposureOpKernel(px, params);
  px        = GPU_ContrastOpKernel(px, params);
  px        = GPU_ToneOpKernel(px, params);
  px        = GPU_HighlightOpKernel(px, params);
  px        = GPU_ShadowOpKernel(px, params);
  px        = GPU_CurveOpKernel(px, params);
  px        = GPU_SaturationOpKernel(px, params);
  px        = GPU_VibranceOpKernel(px, params);
  px        = GPU_ColorWheelOpKernel(px, params);
  px        = GPU_HLSOpKernel(px, params);
  px        = GPU_LMT_Kernel(px, params, lmt_lut);
  px        = GPU_OUTPUT_Kernel(px, params);

  (void)lmt_lut;
  dst.write(px, gid);
}
