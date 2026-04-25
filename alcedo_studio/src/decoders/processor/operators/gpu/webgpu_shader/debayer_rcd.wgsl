//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

const kEps: f32 = 1e-5;
const kEpsSq: f32 = 1e-10;

struct SinglePlaneParams {
  rgb_fc: vec4<u32>,
  width: u32,
  height: u32,
  stride: u32,
  padding: u32,
};

struct MergeParams {
  width: u32,
  height: u32,
  plane_stride: u32,
  rgba_stride: u32,
};

fn FC(params: SinglePlaneParams, y: u32, x: u32) -> u32 {
  let idx = ((y & 1u) << 1u) | (x & 1u);
  return params.rgb_fc[idx];
}

fn ReconstructRbAtGreen(
  ch_m1: f32, ch_p1: f32, ch_m3: f32, ch_p3: f32,
  ch_l1: f32, ch_r1: f32, ch_l3: f32, ch_r3: f32,
  g_c: f32, g_m2: f32, g_p2: f32, g_l2: f32, g_r2: f32,
  g_m1: f32, g_p1: f32, g_l1: f32, g_r1: f32,
  vh_disc: f32
) -> f32 {
  let N_grad = kEps + abs(g_c - g_m2) + abs(ch_m1 - ch_p1) + abs(ch_m1 - ch_m3);
  let S_grad = kEps + abs(g_c - g_p2) + abs(ch_p1 - ch_m1) + abs(ch_p1 - ch_p3);
  let W_grad = kEps + abs(g_c - g_l2) + abs(ch_l1 - ch_r1) + abs(ch_l1 - ch_l3);
  let E_grad = kEps + abs(g_c - g_r2) + abs(ch_r1 - ch_l1) + abs(ch_r1 - ch_r3);

  let N_est = ch_m1 - g_m1;
  let S_est = ch_p1 - g_p1;
  let W_est = ch_l1 - g_l1;
  let E_est = ch_r1 - g_r1;

  let V_est = (N_grad * S_est + S_grad * N_est) / (N_grad + S_grad);
  let H_est = (E_grad * W_est + W_grad * E_est) / (E_grad + W_grad);
  return max(0.0, g_c + (1.0 - vh_disc) * V_est + vh_disc * H_est);
}

// ---------------------------------------------------------------------------
// Kernel 1: rcd_init_and_vh
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> raw_init: array<f32>;
@group(0) @binding(1) var<storage, read_write> r_init: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_init: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_init: array<f32>;
@group(0) @binding(4) var<storage, read_write> vh_dir_init: array<f32>;
@group(0) @binding(5) var<uniform> params_init: SinglePlaneParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_init_and_vh(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_init.width || gid.y >= params_init.height) {
    return;
  }

  let index = gid.y * params_init.stride + gid.x;
  let val = raw_init[index];
  let color = FC(params_init, gid.y, gid.x);

  r_init[index] = select(0.0, val, color == 0u);
  g_init[index] = select(0.0, val, color == 1u);
  b_init[index] = select(0.0, val, color == 2u);

  var vh = 0.0;
  if (gid.x >= 4u && gid.y >= 4u && gid.x + 4u < params_init.width && gid.y + 4u < params_init.height) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = val;

    let vm1 = raw_init[u32(y - 1) * params_init.stride + u32(x)];
    let vp1 = raw_init[u32(y + 1) * params_init.stride + u32(x)];
    let vm2 = raw_init[u32(y - 2) * params_init.stride + u32(x)];
    let vp2 = raw_init[u32(y + 2) * params_init.stride + u32(x)];
    let vm3 = raw_init[u32(y - 3) * params_init.stride + u32(x)];
    let vp3 = raw_init[u32(y + 3) * params_init.stride + u32(x)];
    let vm4 = raw_init[u32(y - 4) * params_init.stride + u32(x)];
    let vp4 = raw_init[u32(y + 4) * params_init.stride + u32(x)];

    let hm1 = raw_init[u32(y) * params_init.stride + u32(x - 1)];
    let hp1 = raw_init[u32(y) * params_init.stride + u32(x + 1)];
    let hm2 = raw_init[u32(y) * params_init.stride + u32(x - 2)];
    let hp2 = raw_init[u32(y) * params_init.stride + u32(x + 2)];
    let hm3 = raw_init[u32(y) * params_init.stride + u32(x - 3)];
    let hp3 = raw_init[u32(y) * params_init.stride + u32(x + 3)];
    let hm4 = raw_init[u32(y) * params_init.stride + u32(x - 4)];
    let hp4 = raw_init[u32(y) * params_init.stride + u32(x + 4)];

    let V_stat = max(
      -18.0 * c * vm1 - 18.0 * c * vp1 - 36.0 * c * vm2 - 36.0 * c * vp2 + 18.0 * c * vm3 +
      18.0 * c * vp3 - 2.0 * c * vm4 - 2.0 * c * vp4 + 38.0 * c * c - 70.0 * vm1 * vp1 -
      12.0 * vm1 * vm2 + 24.0 * vm1 * vp2 - 38.0 * vm1 * vm3 + 16.0 * vm1 * vp3 +
      12.0 * vm1 * vm4 - 6.0 * vm1 * vp4 + 46.0 * vm1 * vm1 + 24.0 * vp1 * vm2 -
      12.0 * vp1 * vp2 + 16.0 * vp1 * vm3 - 38.0 * vp1 * vp3 - 6.0 * vp1 * vm4 +
      12.0 * vp1 * vp4 + 46.0 * vp1 * vp1 + 14.0 * vm2 * vp2 - 12.0 * vm2 * vp3 -
      2.0 * vm2 * vm4 + 2.0 * vm2 * vp4 + 11.0 * vm2 * vm2 - 12.0 * vp2 * vm3 +
      2.0 * vp2 * vm4 - 2.0 * vp2 * vp4 + 11.0 * vp2 * vp2 + 2.0 * vm3 * vp3 -
      6.0 * vm3 * vm4 + 10.0 * vm3 * vm3 - 6.0 * vp3 * vp4 + 10.0 * vp3 * vp3 +
      1.0 * vm4 * vm4 + 1.0 * vp4 * vp4,
      kEpsSq
    );

    let H_stat = max(
      -18.0 * c * hm1 - 18.0 * c * hp1 - 36.0 * c * hm2 - 36.0 * c * hp2 + 18.0 * c * hm3 +
      18.0 * c * hp3 - 2.0 * c * hm4 - 2.0 * c * hp4 + 38.0 * c * c - 70.0 * hm1 * hp1 -
      12.0 * hm1 * hm2 + 24.0 * hm1 * hp2 - 38.0 * hm1 * hm3 + 16.0 * hm1 * hp3 +
      12.0 * hm1 * hm4 - 6.0 * hm1 * hp4 + 46.0 * hm1 * hm1 + 24.0 * hp1 * hm2 -
      12.0 * hp1 * hp2 + 16.0 * hp1 * hm3 - 38.0 * hp1 * hp3 - 6.0 * hp1 * hm4 +
      12.0 * hp1 * hp4 + 46.0 * hp1 * hp1 + 14.0 * hm2 * hp2 - 12.0 * hm2 * hp3 -
      2.0 * hm2 * hm4 + 2.0 * hm2 * hp4 + 11.0 * hm2 * hm2 - 12.0 * hp2 * hm3 +
      2.0 * hp2 * hm4 - 2.0 * hp2 * hp4 + 11.0 * hp2 * hp2 + 2.0 * hm3 * hp3 -
      6.0 * hm3 * hm4 + 10.0 * hm3 * hm3 - 6.0 * hp3 * hp4 + 10.0 * hp3 * hp3 +
      1.0 * hm4 * hm4 + 1.0 * hp4 * hp4,
      kEpsSq
    );

    vh = V_stat / (V_stat + H_stat);
  }

  vh_dir_init[index] = vh;
}

// ---------------------------------------------------------------------------
// Kernel 2: rcd_green_at_rb
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> raw_green: array<f32>;
@group(0) @binding(1) var<storage, read> vh_dir_green: array<f32>;
@group(0) @binding(2) var<storage, read_write> g_green: array<f32>;
@group(0) @binding(3) var<uniform> params_green: SinglePlaneParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_green_at_rb(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_green.width || gid.y >= params_green.height || gid.x < 4u || gid.y < 4u ||
      gid.x + 4u >= params_green.width || gid.y + 4u >= params_green.height) {
    return;
  }

  if (FC(params_green, gid.y, gid.x) == 1u) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);

  let VH_central = vh_dir_green[u32(y) * params_green.stride + u32(x)];
  let VH_neigh = 0.25 * (vh_dir_green[u32(y - 1) * params_green.stride + u32(x - 1)] +
                         vh_dir_green[u32(y - 1) * params_green.stride + u32(x + 1)] +
                         vh_dir_green[u32(y + 1) * params_green.stride + u32(x - 1)] +
                         vh_dir_green[u32(y + 1) * params_green.stride + u32(x + 1)]);
  let VH_disc = select(VH_central, VH_neigh, abs(0.5 - VH_central) < abs(0.5 - VH_neigh));

  let c = raw_green[u32(y) * params_green.stride + u32(x)];
  let vm1 = raw_green[u32(y - 1) * params_green.stride + u32(x)];
  let vp1 = raw_green[u32(y + 1) * params_green.stride + u32(x)];
  let vm2 = raw_green[u32(y - 2) * params_green.stride + u32(x)];
  let vp2 = raw_green[u32(y + 2) * params_green.stride + u32(x)];
  let vm3 = raw_green[u32(y - 3) * params_green.stride + u32(x)];
  let vp3 = raw_green[u32(y + 3) * params_green.stride + u32(x)];
  let vm4 = raw_green[u32(y - 4) * params_green.stride + u32(x)];
  let vp4 = raw_green[u32(y + 4) * params_green.stride + u32(x)];

  let hm1 = raw_green[u32(y) * params_green.stride + u32(x - 1)];
  let hp1 = raw_green[u32(y) * params_green.stride + u32(x + 1)];
  let hm2 = raw_green[u32(y) * params_green.stride + u32(x - 2)];
  let hp2 = raw_green[u32(y) * params_green.stride + u32(x + 2)];
  let hm3 = raw_green[u32(y) * params_green.stride + u32(x - 3)];
  let hp3 = raw_green[u32(y) * params_green.stride + u32(x + 3)];
  let hm4 = raw_green[u32(y) * params_green.stride + u32(x - 4)];
  let hp4 = raw_green[u32(y) * params_green.stride + u32(x + 4)];

  let lpf_c = 0.25 * c + 0.125 * (vm1 + vp1 + hm1 + hp1) +
    0.0625 * (raw_green[u32(y - 1) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y - 1) * params_green.stride + u32(x + 1)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x + 1)]);

  let lpf_n2 = 0.25 * vm2 + 0.125 * (raw_green[u32(y - 3) * params_green.stride + u32(x)] + c +
                                      raw_green[u32(y - 2) * params_green.stride + u32(x - 1)] +
                                      raw_green[u32(y - 2) * params_green.stride + u32(x + 1)]) +
    0.0625 * (raw_green[u32(y - 3) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y - 3) * params_green.stride + u32(x + 1)] +
              raw_green[u32(y - 1) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y - 1) * params_green.stride + u32(x + 1)]);

  let lpf_s2 = 0.25 * vp2 + 0.125 * (c + raw_green[u32(y + 3) * params_green.stride + u32(x)] +
                                      raw_green[u32(y + 2) * params_green.stride + u32(x - 1)] +
                                      raw_green[u32(y + 2) * params_green.stride + u32(x + 1)]) +
    0.0625 * (raw_green[u32(y + 1) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x + 1)] +
              raw_green[u32(y + 3) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y + 3) * params_green.stride + u32(x + 1)]);

  let lpf_w2 = 0.25 * hm2 + 0.125 * (raw_green[u32(y) * params_green.stride + u32(x - 3)] + c +
                                      raw_green[u32(y - 1) * params_green.stride + u32(x - 2)] +
                                      raw_green[u32(y + 1) * params_green.stride + u32(x - 2)]) +
    0.0625 * (raw_green[u32(y - 1) * params_green.stride + u32(x - 3)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x - 3)] +
              raw_green[u32(y - 1) * params_green.stride + u32(x - 1)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x - 1)]);

  let lpf_e2 = 0.25 * hp2 + 0.125 * (c + raw_green[u32(y) * params_green.stride + u32(x + 3)] +
                                      raw_green[u32(y - 1) * params_green.stride + u32(x + 2)] +
                                      raw_green[u32(y + 1) * params_green.stride + u32(x + 2)]) +
    0.0625 * (raw_green[u32(y - 1) * params_green.stride + u32(x + 1)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x + 1)] +
              raw_green[u32(y - 1) * params_green.stride + u32(x + 3)] +
              raw_green[u32(y + 1) * params_green.stride + u32(x + 3)]);

  let N_grad = kEps + abs(vm1 - vp1) + abs(c - vm2) + abs(vm1 - vm3) + abs(vm2 - vm4);
  let S_grad = kEps + abs(vp1 - vm1) + abs(c - vp2) + abs(vp1 - vp3) + abs(vp2 - vp4);
  let W_grad = kEps + abs(hm1 - hp1) + abs(c - hm2) + abs(hm1 - hm3) + abs(hm2 - hm4);
  let E_grad = kEps + abs(hp1 - hm1) + abs(c - hp2) + abs(hp1 - hp3) + abs(hp2 - hp4);

  let N_est = vm1 * (1.0 + (lpf_c - lpf_n2) / (kEps + lpf_c + lpf_n2));
  let S_est = vp1 * (1.0 + (lpf_c - lpf_s2) / (kEps + lpf_c + lpf_s2));
  let W_est = hm1 * (1.0 + (lpf_c - lpf_w2) / (kEps + lpf_c + lpf_w2));
  let E_est = hp1 * (1.0 + (lpf_c - lpf_e2) / (kEps + lpf_c + lpf_e2));

  let V_est = (S_grad * N_est + N_grad * S_est) / (N_grad + S_grad);
  let H_est = (W_grad * E_est + E_grad * W_est) / (E_grad + W_grad);

  g_green[gid.y * params_green.stride + gid.x] = max(VH_disc * H_est + (1.0 - VH_disc) * V_est, 0.0);
}

// ---------------------------------------------------------------------------
// Kernel 3: rcd_pq_dir
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> raw_pq: array<f32>;
@group(0) @binding(1) var<storage, read_write> pq_dir_pq: array<f32>;
@group(0) @binding(2) var<uniform> params_pq: SinglePlaneParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_pq_dir(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_pq.width || gid.y >= params_pq.height) {
    return;
  }

  var pq = 0.0;
  if (gid.x >= 4u && gid.y >= 4u && gid.x + 4u < params_pq.width && gid.y + 4u < params_pq.height &&
      FC(params_pq, gid.y, gid.x) != 1u) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = raw_pq[u32(y) * params_pq.stride + u32(x)];

    let nw1 = raw_pq[u32(y - 1) * params_pq.stride + u32(x - 1)];
    let se1 = raw_pq[u32(y + 1) * params_pq.stride + u32(x + 1)];
    let nw2 = raw_pq[u32(y - 2) * params_pq.stride + u32(x - 2)];
    let se2 = raw_pq[u32(y + 2) * params_pq.stride + u32(x + 2)];
    let nw3 = raw_pq[u32(y - 3) * params_pq.stride + u32(x - 3)];
    let se3 = raw_pq[u32(y + 3) * params_pq.stride + u32(x + 3)];
    let nw4 = raw_pq[u32(y - 4) * params_pq.stride + u32(x - 4)];
    let se4 = raw_pq[u32(y + 4) * params_pq.stride + u32(x + 4)];

    let sw1 = raw_pq[u32(y + 1) * params_pq.stride + u32(x - 1)];
    let ne1 = raw_pq[u32(y - 1) * params_pq.stride + u32(x + 1)];
    let sw2 = raw_pq[u32(y + 2) * params_pq.stride + u32(x - 2)];
    let ne2 = raw_pq[u32(y - 2) * params_pq.stride + u32(x + 2)];
    let sw3 = raw_pq[u32(y + 3) * params_pq.stride + u32(x - 3)];
    let ne3 = raw_pq[u32(y - 3) * params_pq.stride + u32(x + 3)];
    let sw4 = raw_pq[u32(y + 4) * params_pq.stride + u32(x - 4)];
    let ne4 = raw_pq[u32(y - 4) * params_pq.stride + u32(x + 4)];

    let P_stat = max(
      -18.0 * c * nw1 - 18.0 * c * se1 - 36.0 * c * nw2 - 36.0 * c * se2 + 18.0 * c * nw3 +
      18.0 * c * se3 - 2.0 * c * nw4 - 2.0 * c * se4 + 38.0 * c * c - 70.0 * nw1 * se1 -
      12.0 * nw1 * nw2 + 24.0 * nw1 * se2 - 38.0 * nw1 * nw3 + 16.0 * nw1 * se3 +
      12.0 * nw1 * nw4 - 6.0 * nw1 * se4 + 46.0 * nw1 * nw1 + 24.0 * se1 * nw2 -
      12.0 * se1 * se2 + 16.0 * se1 * nw3 - 38.0 * se1 * se3 - 6.0 * se1 * nw4 +
      12.0 * se1 * se4 + 46.0 * se1 * se1 + 14.0 * nw2 * se2 - 12.0 * nw2 * se3 -
      2.0 * nw2 * nw4 + 2.0 * nw2 * se4 + 11.0 * nw2 * nw2 - 12.0 * se2 * nw3 +
      2.0 * se2 * nw4 - 2.0 * se2 * se4 + 11.0 * se2 * se2 + 2.0 * nw3 * se3 -
      6.0 * nw3 * nw4 + 10.0 * nw3 * nw3 - 6.0 * se3 * se4 + 10.0 * se3 * se3 +
      1.0 * nw4 * nw4 + 1.0 * se4 * se4,
      kEpsSq
    );

    let Q_stat = max(
      -18.0 * c * sw1 - 18.0 * c * ne1 - 36.0 * c * sw2 - 36.0 * c * ne2 + 18.0 * c * sw3 +
      18.0 * c * ne3 - 2.0 * c * sw4 - 2.0 * c * ne4 + 38.0 * c * c - 70.0 * sw1 * ne1 -
      12.0 * sw1 * sw2 + 24.0 * sw1 * ne2 - 38.0 * sw1 * sw3 + 16.0 * sw1 * ne3 +
      12.0 * sw1 * sw4 - 6.0 * sw1 * ne4 + 46.0 * sw1 * sw1 + 24.0 * ne1 * sw2 -
      12.0 * ne1 * ne2 + 16.0 * ne1 * sw3 - 38.0 * ne1 * ne3 - 6.0 * ne1 * sw4 +
      12.0 * ne1 * ne4 + 46.0 * ne1 * ne1 + 14.0 * sw2 * ne2 - 12.0 * sw2 * ne3 -
      2.0 * sw2 * sw4 + 2.0 * sw2 * ne4 + 11.0 * sw2 * sw2 - 12.0 * ne2 * sw3 +
      2.0 * ne2 * sw4 - 2.0 * ne2 * ne4 + 11.0 * ne2 * ne2 + 2.0 * sw3 * ne3 -
      6.0 * sw3 * sw4 + 10.0 * sw3 * sw3 - 6.0 * ne3 * ne4 + 10.0 * ne3 * ne3 +
      1.0 * sw4 * sw4 + 1.0 * ne4 * ne4,
      kEpsSq
    );

    pq = P_stat / (P_stat + Q_stat);
  }

  pq_dir_pq[gid.y * params_pq.stride + gid.x] = pq;
}

// ---------------------------------------------------------------------------
// Kernel 4: rcd_rb_at_rb
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> pq_dir_rb: array<f32>;
@group(0) @binding(1) var<storage, read> g_rb: array<f32>;
@group(0) @binding(2) var<storage, read_write> r_rb: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_rb: array<f32>;
@group(0) @binding(4) var<uniform> params_rb: SinglePlaneParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_rb_at_rb(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_rb.width || gid.y >= params_rb.height || gid.x < 4u || gid.y < 4u ||
      gid.x + 4u >= params_rb.width || gid.y + 4u >= params_rb.height) {
    return;
  }

  let color = FC(params_rb, gid.y, gid.x);
  if (color == 1u) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let c = 2u - color;

  let PQ_c = pq_dir_rb[u32(y) * params_rb.stride + u32(x)];
  let PQ_n = 0.25 * (pq_dir_rb[u32(y - 1) * params_rb.stride + u32(x - 1)] +
                     pq_dir_rb[u32(y - 1) * params_rb.stride + u32(x + 1)] +
                     pq_dir_rb[u32(y + 1) * params_rb.stride + u32(x - 1)] +
                     pq_dir_rb[u32(y + 1) * params_rb.stride + u32(x + 1)]);
  let PQ_disc = select(PQ_c, PQ_n, abs(0.5 - PQ_c) < abs(0.5 - PQ_n));

  let g_c = g_rb[u32(y) * params_rb.stride + u32(x)];

  let ch_nw1 = select(r_rb[u32(y - 1) * params_rb.stride + u32(x - 1)],
                      b_rb[u32(y - 1) * params_rb.stride + u32(x - 1)], c == 1u);
  let ch_ne1 = select(r_rb[u32(y - 1) * params_rb.stride + u32(x + 1)],
                      b_rb[u32(y - 1) * params_rb.stride + u32(x + 1)], c == 1u);
  let ch_sw1 = select(r_rb[u32(y + 1) * params_rb.stride + u32(x - 1)],
                      b_rb[u32(y + 1) * params_rb.stride + u32(x - 1)], c == 1u);
  let ch_se1 = select(r_rb[u32(y + 1) * params_rb.stride + u32(x + 1)],
                      b_rb[u32(y + 1) * params_rb.stride + u32(x + 1)], c == 1u);

  let ch_nw3 = select(r_rb[u32(y - 3) * params_rb.stride + u32(x - 3)],
                      b_rb[u32(y - 3) * params_rb.stride + u32(x - 3)], c == 1u);
  let ch_ne3 = select(r_rb[u32(y - 3) * params_rb.stride + u32(x + 3)],
                      b_rb[u32(y - 3) * params_rb.stride + u32(x + 3)], c == 1u);
  let ch_sw3 = select(r_rb[u32(y + 3) * params_rb.stride + u32(x - 3)],
                      b_rb[u32(y + 3) * params_rb.stride + u32(x - 3)], c == 1u);
  let ch_se3 = select(r_rb[u32(y + 3) * params_rb.stride + u32(x + 3)],
                      b_rb[u32(y + 3) * params_rb.stride + u32(x + 3)], c == 1u);

  let g_nw2 = g_rb[u32(y - 2) * params_rb.stride + u32(x - 2)];
  let g_ne2 = g_rb[u32(y - 2) * params_rb.stride + u32(x + 2)];
  let g_sw2 = g_rb[u32(y + 2) * params_rb.stride + u32(x - 2)];
  let g_se2 = g_rb[u32(y + 2) * params_rb.stride + u32(x + 2)];

  let NW_grad = kEps + abs(ch_nw1 - ch_se1) + abs(ch_nw1 - ch_nw3) + abs(g_c - g_nw2);
  let NE_grad = kEps + abs(ch_ne1 - ch_sw1) + abs(ch_ne1 - ch_ne3) + abs(g_c - g_ne2);
  let SW_grad = kEps + abs(ch_sw1 - ch_ne1) + abs(ch_sw1 - ch_sw3) + abs(g_c - g_sw2);
  let SE_grad = kEps + abs(ch_se1 - ch_nw1) + abs(ch_se1 - ch_se3) + abs(g_c - g_se2);

  let g_nw1 = g_rb[u32(y - 1) * params_rb.stride + u32(x - 1)];
  let g_ne1 = g_rb[u32(y - 1) * params_rb.stride + u32(x + 1)];
  let g_sw1 = g_rb[u32(y + 1) * params_rb.stride + u32(x - 1)];
  let g_se1 = g_rb[u32(y + 1) * params_rb.stride + u32(x + 1)];

  let NW_est = ch_nw1 - g_nw1;
  let NE_est = ch_ne1 - g_ne1;
  let SW_est = ch_sw1 - g_sw1;
  let SE_est = ch_se1 - g_se1;

  let P_est = (NW_grad * SE_est + SE_grad * NW_est) / (NW_grad + SE_grad);
  let Q_est = (NE_grad * SW_est + SW_grad * NE_est) / (NE_grad + SW_grad);
  let out_val = max(0.0, g_c + (1.0 - PQ_disc) * P_est + PQ_disc * Q_est);

  if (c == 0u) {
    r_rb[gid.y * params_rb.stride + gid.x] = out_val;
  } else {
    b_rb[gid.y * params_rb.stride + gid.x] = out_val;
  }
}

// ---------------------------------------------------------------------------
// Kernel 5: rcd_rb_at_g
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> vh_dir_g: array<f32>;
@group(0) @binding(1) var<storage, read> g_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> r_g: array<f32>;
@group(0) @binding(3) var<storage, read_write> b_g: array<f32>;
@group(0) @binding(4) var<uniform> params_g: SinglePlaneParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_rb_at_g(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_g.width || gid.y >= params_g.height || gid.x < 4u || gid.y < 4u ||
      gid.x + 4u >= params_g.width || gid.y + 4u >= params_g.height ||
      FC(params_g, gid.y, gid.x) != 1u) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);

  let VH_central = vh_dir_g[u32(y) * params_g.stride + u32(x)];
  let VH_neigh = 0.25 * (vh_dir_g[u32(y - 1) * params_g.stride + u32(x - 1)] +
                         vh_dir_g[u32(y - 1) * params_g.stride + u32(x + 1)] +
                         vh_dir_g[u32(y + 1) * params_g.stride + u32(x - 1)] +
                         vh_dir_g[u32(y + 1) * params_g.stride + u32(x + 1)]);
  let VH_disc = select(VH_central, VH_neigh, abs(0.5 - VH_central) < abs(0.5 - VH_neigh));

  let g_c = g_g[u32(y) * params_g.stride + u32(x)];
  let g_m2 = g_g[u32(y - 2) * params_g.stride + u32(x)];
  let g_p2 = g_g[u32(y + 2) * params_g.stride + u32(x)];
  let g_l2 = g_g[u32(y) * params_g.stride + u32(x - 2)];
  let g_r2 = g_g[u32(y) * params_g.stride + u32(x + 2)];
  let g_m1 = g_g[u32(y - 1) * params_g.stride + u32(x)];
  let g_p1 = g_g[u32(y + 1) * params_g.stride + u32(x)];
  let g_l1 = g_g[u32(y) * params_g.stride + u32(x - 1)];
  let g_r1 = g_g[u32(y) * params_g.stride + u32(x + 1)];

  let r_m1 = r_g[u32(y - 1) * params_g.stride + u32(x)];
  let r_p1 = r_g[u32(y + 1) * params_g.stride + u32(x)];
  let r_m3 = r_g[u32(y - 3) * params_g.stride + u32(x)];
  let r_p3 = r_g[u32(y + 3) * params_g.stride + u32(x)];
  let r_l1 = r_g[u32(y) * params_g.stride + u32(x - 1)];
  let r_r1 = r_g[u32(y) * params_g.stride + u32(x + 1)];
  let r_l3 = r_g[u32(y) * params_g.stride + u32(x - 3)];
  let r_r3 = r_g[u32(y) * params_g.stride + u32(x + 3)];

  let b_m1 = b_g[u32(y - 1) * params_g.stride + u32(x)];
  let b_p1 = b_g[u32(y + 1) * params_g.stride + u32(x)];
  let b_m3 = b_g[u32(y - 3) * params_g.stride + u32(x)];
  let b_p3 = b_g[u32(y + 3) * params_g.stride + u32(x)];
  let b_l1 = b_g[u32(y) * params_g.stride + u32(x - 1)];
  let b_r1 = b_g[u32(y) * params_g.stride + u32(x + 1)];
  let b_l3 = b_g[u32(y) * params_g.stride + u32(x - 3)];
  let b_r3 = b_g[u32(y) * params_g.stride + u32(x + 3)];

  let index = gid.y * params_g.stride + gid.x;

  r_g[index] = ReconstructRbAtGreen(
    r_m1, r_p1, r_m3, r_p3, r_l1, r_r1, r_l3, r_r3,
    g_c, g_m2, g_p2, g_l2, g_r2, g_m1, g_p1, g_l1, g_r1, VH_disc
  );
  b_g[index] = ReconstructRbAtGreen(
    b_m1, b_p1, b_m3, b_p3, b_l1, b_r1, b_l3, b_r3,
    g_c, g_m2, g_p2, g_l2, g_r2, g_m1, g_p1, g_l1, g_r1, VH_disc
  );
}

// ---------------------------------------------------------------------------
// Kernel 6: rcd_merge_rgba
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> r_merge: array<f32>;
@group(0) @binding(1) var<storage, read> g_merge: array<f32>;
@group(0) @binding(2) var<storage, read> b_merge: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_rgba: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params_merge: MergeParams;

@compute @workgroup_size(8, 8, 1)
fn rcd_merge_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_merge.width || gid.y >= params_merge.height) {
    return;
  }

  let plane_index = gid.y * params_merge.plane_stride + gid.x;
  let rgba_index = gid.y * params_merge.rgba_stride + gid.x;
  out_rgba[rgba_index] = vec4<f32>(
    r_merge[plane_index],
    g_merge[plane_index],
    b_merge[plane_index],
    1.0
  );
}
