//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <metal_stdlib>

using namespace metal;

struct ResizeParams {
  uint  origin_x;
  uint  origin_y;
  uint  crop_width;
  uint  crop_height;
  uint  dst_width;
  uint  dst_height;
  uint  src_stride;
  uint  dst_stride;
  float scale_x;
  float scale_y;
};

struct AffineParams {
  float  m00;
  float  m01;
  float  m02;
  float  m10;
  float  m11;
  float  m12;
  uint   src_width;
  uint   src_height;
  uint   dst_width;
  uint   dst_height;
  uint   src_stride;
  uint   dst_stride;
  float4 border;
};

template <typename PixelT>
struct PixelOps;

template <>
struct PixelOps<float> {
  using Acc = float;

  static inline auto Zero() -> Acc { return 0.0f; }
  static inline auto AddMul(Acc acc, float value, float weight) -> Acc {
    return fma(value, weight, acc);
  }
  static inline auto Div(Acc acc, float denom) -> float { return acc / denom; }
};

template <>
struct PixelOps<float4> {
  using Acc = float4;

  static inline auto Zero() -> Acc { return float4(0.0f); }
  static inline auto AddMul(Acc acc, float4 value, float weight) -> Acc {
    return fma(value, float4(weight), acc);
  }
  static inline auto Div(Acc acc, float denom) -> float4 { return acc / denom; }
};

template <typename PixelT>
static inline auto ZeroPixel() -> PixelT {
  return PixelT(0.0f);
}

template <typename PixelT>
static inline auto ReadWithinCrop(device const PixelT* src, constant ResizeParams& params, int x,
                                  int y) -> PixelT {
  const int crop_x0 = static_cast<int>(params.origin_x);
  const int crop_y0 = static_cast<int>(params.origin_y);
  const int crop_x1 = static_cast<int>(params.origin_x + params.crop_width);
  const int crop_y1 = static_cast<int>(params.origin_y + params.crop_height);
  if (x < crop_x0 || y < crop_y0 || x >= crop_x1 || y >= crop_y1) {
    return ZeroPixel<PixelT>();
  }
  return src[static_cast<uint>(y) * params.src_stride + static_cast<uint>(x)];
}

template <typename PixelT>
static inline auto BilinearSampleWithinCrop(device const PixelT* src, constant ResizeParams& params,
                                            float sx, float sy) -> PixelT {
  using Ops = PixelOps<PixelT>;
  using Acc = typename Ops::Acc;

  const int x0 = static_cast<int>(floor(sx));
  const int y0 = static_cast<int>(floor(sy));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);

  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx * fy;

  const PixelT p00 = ReadWithinCrop(src, params, x0, y0);
  const PixelT p10 = ReadWithinCrop(src, params, x1, y0);
  const PixelT p01 = ReadWithinCrop(src, params, x0, y1);
  const PixelT p11 = ReadWithinCrop(src, params, x1, y1);

  Acc acc = Ops::Zero();
  acc     = Ops::AddMul(acc, p00, w00);
  acc     = Ops::AddMul(acc, p10, w10);
  acc     = Ops::AddMul(acc, p01, w01);
  acc     = Ops::AddMul(acc, p11, w11);
  return Ops::Div(acc, 1.0f);
}

template <typename PixelT>
static inline auto BorderValue(constant AffineParams& params) -> PixelT;

template <>
inline auto BorderValue<float>(constant AffineParams& params) -> float {
  return params.border.x;
}

template <>
inline auto BorderValue<float4>(constant AffineParams& params) -> float4 {
  return params.border;
}

template <typename PixelT>
static inline auto ReadOrBorder(device const PixelT* src, constant AffineParams& params, int x, int y)
    -> PixelT {
  if (x < 0 || y < 0 || x >= static_cast<int>(params.src_width) ||
      y >= static_cast<int>(params.src_height)) {
    return BorderValue<PixelT>(params);
  }
  return src[static_cast<uint>(y) * params.src_stride + static_cast<uint>(x)];
}

template <typename PixelT>
static inline auto BilinearSampleAffine(device const PixelT* src, constant AffineParams& params,
                                        float sx, float sy) -> PixelT {
  using Ops = PixelOps<PixelT>;
  using Acc = typename Ops::Acc;

  const int x0 = static_cast<int>(floor(sx));
  const int y0 = static_cast<int>(floor(sy));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const float fx = sx - static_cast<float>(x0);
  const float fy = sy - static_cast<float>(y0);

  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx * fy;

  const PixelT p00 = ReadOrBorder(src, params, x0, y0);
  const PixelT p10 = ReadOrBorder(src, params, x1, y0);
  const PixelT p01 = ReadOrBorder(src, params, x0, y1);
  const PixelT p11 = ReadOrBorder(src, params, x1, y1);

  Acc acc = Ops::Zero();
  acc     = Ops::AddMul(acc, p00, w00);
  acc     = Ops::AddMul(acc, p10, w10);
  acc     = Ops::AddMul(acc, p01, w01);
  acc     = Ops::AddMul(acc, p11, w11);
  return Ops::Div(acc, 1.0f);
}

template <typename PixelT>
static inline void CropResizeLinear(device const PixelT* src, device PixelT* dst,
                                    constant ResizeParams& params, uint2 gid) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const float sx = static_cast<float>(params.origin_x) +
                   (static_cast<float>(gid.x) + 0.5f) * params.scale_x - 0.5f;
  const float sy = static_cast<float>(params.origin_y) +
                   (static_cast<float>(gid.y) + 0.5f) * params.scale_y - 0.5f;
  dst[gid.y * params.dst_stride + gid.x] = BilinearSampleWithinCrop(src, params, sx, sy);
}

template <typename PixelT>
static inline void CropResizeArea(device const PixelT* src, device PixelT* dst,
                                  constant ResizeParams& params, uint2 gid) {
  using Ops = PixelOps<PixelT>;
  using Acc = typename Ops::Acc;

  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  constexpr float kEps = 1e-8f;

  const float sx0 =
      static_cast<float>(params.origin_x) + static_cast<float>(gid.x) * params.scale_x;
  const float sx1 =
      static_cast<float>(params.origin_x) + static_cast<float>(gid.x + 1u) * params.scale_x;
  const float sy0 =
      static_cast<float>(params.origin_y) + static_cast<float>(gid.y) * params.scale_y;
  const float sy1 =
      static_cast<float>(params.origin_y) + static_cast<float>(gid.y + 1u) * params.scale_y;

  const int crop_x0 = static_cast<int>(params.origin_x);
  const int crop_y0 = static_cast<int>(params.origin_y);
  const int crop_x1 = static_cast<int>(params.origin_x + params.crop_width);
  const int crop_y1 = static_cast<int>(params.origin_y + params.crop_height);

  const int ix0 = max(crop_x0, static_cast<int>(floor(sx0)));
  const int ix1 = min(crop_x1, static_cast<int>(ceil(sx1)));
  const int iy0 = max(crop_y0, static_cast<int>(floor(sy0)));
  const int iy1 = min(crop_y1, static_cast<int>(ceil(sy1)));

  Acc   acc   = Ops::Zero();
  float total = 0.0f;

  for (int yy = iy0; yy < iy1; ++yy) {
    const float yy0 = max(sy0, static_cast<float>(yy));
    const float yy1 = min(sy1, static_cast<float>(yy + 1));
    const float wy  = max(0.0f, yy1 - yy0);
    if (wy <= 0.0f) {
      continue;
    }

    for (int xx = ix0; xx < ix1; ++xx) {
      const float xx0 = max(sx0, static_cast<float>(xx));
      const float xx1 = min(sx1, static_cast<float>(xx + 1));
      const float wx  = max(0.0f, xx1 - xx0);
      const float w   = wx * wy;
      if (w <= 0.0f) {
        continue;
      }

      acc =
          Ops::AddMul(acc, src[static_cast<uint>(yy) * params.src_stride + static_cast<uint>(xx)], w);
      total += w;
    }
  }

  const uint dst_index = gid.y * params.dst_stride + gid.x;
  if (total <= kEps) {
    const int sx = clamp(static_cast<int>(sx0), crop_x0, crop_x1 - 1);
    const int sy = clamp(static_cast<int>(sy0), crop_y0, crop_y1 - 1);
    dst[dst_index] = src[static_cast<uint>(sy) * params.src_stride + static_cast<uint>(sx)];
    return;
  }

  dst[dst_index] = Ops::Div(acc, total);
}

template <typename PixelT>
static inline void WarpAffineLinear(device const PixelT* src, device PixelT* dst,
                                    constant AffineParams& params, uint2 gid) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }

  const float sx = params.m00 * static_cast<float>(gid.x) +
                   params.m01 * static_cast<float>(gid.y) + params.m02;
  const float sy = params.m10 * static_cast<float>(gid.x) +
                   params.m11 * static_cast<float>(gid.y) + params.m12;
  dst[gid.y * params.dst_stride + gid.x] = BilinearSampleAffine(src, params, sx, sy);
}

kernel void crop_resize_linear_r32f(device const float* src [[buffer(0)]],
                                    device float*       dst [[buffer(1)]],
                                    constant ResizeParams& params [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  CropResizeLinear<float>(src, dst, params, gid);
}

kernel void crop_resize_linear_rgba32f(device const float4* src [[buffer(0)]],
                                       device float4*       dst [[buffer(1)]],
                                       constant ResizeParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  CropResizeLinear<float4>(src, dst, params, gid);
}

kernel void crop_resize_area_r32f(device const float* src [[buffer(0)]],
                                  device float*       dst [[buffer(1)]],
                                  constant ResizeParams& params [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  CropResizeArea<float>(src, dst, params, gid);
}

kernel void crop_resize_area_rgba32f(device const float4* src [[buffer(0)]],
                                     device float4*       dst [[buffer(1)]],
                                     constant ResizeParams& params [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  CropResizeArea<float4>(src, dst, params, gid);
}

kernel void warp_affine_linear_r32f(device const float* src [[buffer(0)]],
                                    device float*       dst [[buffer(1)]],
                                    constant AffineParams& params [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  WarpAffineLinear<float>(src, dst, params, gid);
}

kernel void warp_affine_linear_rgba32f(device const float4* src [[buffer(0)]],
                                       device float4*       dst [[buffer(1)]],
                                       constant AffineParams& params [[buffer(2)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  WarpAffineLinear<float4>(src, dst, params, gid);
}
