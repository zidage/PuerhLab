struct WBParams {
  black_level: vec4<f32>,
  white_level: vec4<f32>,
  wb_multipliers: vec4<f32>,
  apply_white_balance: u32,
  padding: vec3<u32>,
};

struct ToLinearRefParams {
  width: u32,
  height: u32,
  stride: u32,
  tile_width: u32,
  tile_height: u32,
  black_tile_width: u32,
  black_tile_height: u32,
  padding: u32,
  raw_fc: array<vec4<u32>, 9>,
};

@group(0) @binding(0) var<storage, read_write> image: array<f32>;
@group(0) @binding(1) var<uniform> params: ToLinearRefParams;
@group(0) @binding(2) var<uniform> wb_params: WBParams;
@group(0) @binding(3) var<storage, read> pattern_black: array<f32>;

fn raw_fc_at(index: u32) -> u32 {
  return params.raw_fc[index / 4u][index % 4u];
}

fn raw_color_at(y: u32, x: u32) -> u32 {
  if (params.tile_width == 0u || params.tile_height == 0u) {
    return 0u;
  }

  let tile_y = y % params.tile_height;
  let tile_x = x % params.tile_width;
  return raw_fc_at(tile_y * params.tile_width + tile_x);
}

fn pattern_black_at(y: u32, x: u32) -> f32 {
  if (params.black_tile_width == 0u || params.black_tile_height == 0u) {
    return 0.0;
  }

  let tile_y = y % params.black_tile_height;
  let tile_x = x % params.black_tile_width;
  return pattern_black[tile_y * params.black_tile_width + tile_x];
}

fn channel_value(values: vec4<f32>, color_idx: u32) -> f32 {
  return values[color_idx];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let color_idx = raw_color_at(gid.y, gid.x);
  let index = gid.y * params.stride + gid.x;
  let sample = image[index];
  let black = channel_value(wb_params.black_level, color_idx) + pattern_black_at(gid.y, gid.x);
  let denom = channel_value(wb_params.white_level, color_idx) - black;
  var pixel_val = 0.0;
  if (denom > 0.0) {
    pixel_val = clamp((sample - black) / denom, 0.0, 1.0);
  }

  if (wb_params.apply_white_balance != 0u && wb_params.wb_multipliers[1] > 0.0 &&
      (color_idx == 0u || color_idx == 2u)) {
    pixel_val *= wb_params.wb_multipliers[color_idx] / wb_params.wb_multipliers[1];
  }

  image[index] = pixel_val;
}
