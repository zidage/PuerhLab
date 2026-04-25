//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

struct ImageParams {
  width: u32,
  height: u32,
  stride: u32,
  channels: u32,
};

struct OrientParams {
  src_width: u32,
  src_height: u32,
  dst_width: u32,
  dst_height: u32,
  src_stride: u32,
  dst_stride: u32,
  flip: u32,
  padding: u32,
  gain: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> clamp_image: array<f32>;
@group(0) @binding(1) var<uniform> clamp_params: ImageParams;

@compute @workgroup_size(8, 8, 1)
fn clamp01(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= clamp_params.width || gid.y >= clamp_params.height) {
    return;
  }

  let base = (gid.y * clamp_params.stride + gid.x) * clamp_params.channels;
  for (var c = 0u; c < clamp_params.channels; c = c + 1u) {
    clamp_image[base + c] = clamp(clamp_image[base + c], 0.0, 1.0);
  }
}

@group(0) @binding(0) var orient_src: texture_2d<f32>;
@group(0) @binding(1) var orient_dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> orient_params: OrientParams;

fn oriented_dst_xy(src_x: u32, src_y: u32) -> vec2<u32> {
  switch orient_params.flip {
    case 3u: {
      return vec2<u32>(orient_params.src_width - 1u - src_x, orient_params.src_height - 1u - src_y);
    }
    case 5u: {
      return vec2<u32>(src_y, orient_params.src_width - 1u - src_x);
    }
    case 6u: {
      return vec2<u32>(orient_params.src_height - 1u - src_y, src_x);
    }
    default: {
      return vec2<u32>(src_x, src_y);
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn apply_inverse_cam_mul_oriented_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= orient_params.src_width || gid.y >= orient_params.src_height) {
    return;
  }

  var rgba = textureLoad(orient_src, vec2<i32>(i32(gid.x), i32(gid.y)), 0) * orient_params.gain;
  rgba.a = 1.0;

  let dst_xy = oriented_dst_xy(gid.x, gid.y);
  textureStore(orient_dst, vec2<i32>(i32(dst_xy.x), i32(dst_xy.y)), rgba);
}
