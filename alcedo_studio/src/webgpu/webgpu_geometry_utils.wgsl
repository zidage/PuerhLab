//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

struct GeoParams {
  src_width: u32,
  src_height: u32,
  dst_width: u32,
  dst_height: u32,
  src_stride: u32,
  dst_stride: u32,
  channels: u32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: GeoParams;

fn CopyPixel(src_idx: u32, dst_idx: u32) {
  if (params.channels == 1u) {
    dst[dst_idx] = src[src_idx];
  } else {
    dst[dst_idx * 4u + 0u] = src[src_idx * 4u + 0u];
    dst[dst_idx * 4u + 1u] = src[src_idx * 4u + 1u];
    dst[dst_idx * 4u + 2u] = src[src_idx * 4u + 2u];
    dst[dst_idx * 4u + 3u] = src[src_idx * 4u + 3u];
  }
}

@compute @workgroup_size(8, 8, 1)
fn rotate_180(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }
  let src_y = params.src_height - 1u - gid.y;
  let src_x = params.src_width - 1u - gid.x;
  CopyPixel(src_y * params.src_stride + src_x, gid.y * params.dst_stride + gid.x);
}

@compute @workgroup_size(8, 8, 1)
fn rotate_90_cw(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }
  let src_y = params.src_height - 1u - gid.x;
  let src_x = gid.y;
  CopyPixel(src_y * params.src_stride + src_x, gid.y * params.dst_stride + gid.x);
}

@compute @workgroup_size(8, 8, 1)
fn rotate_90_ccw(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
    return;
  }
  let src_y = gid.x;
  let src_x = params.src_width - 1u - gid.y;
  CopyPixel(src_y * params.src_stride + src_x, gid.y * params.dst_stride + gid.x);
}
