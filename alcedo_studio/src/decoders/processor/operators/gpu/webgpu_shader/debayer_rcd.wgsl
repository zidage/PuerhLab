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

fn FC(params: SinglePlaneParams, y: u32, x: u32) -> u32 {
  let idx = ((y & 1u) << 1u) | (x & 1u);
  return params.rgb_fc[idx];
}

fn Px(tex: texture_2d<f32>, y: i32, x: i32) -> f32 {
  return textureLoad(tex, vec2<i32>(x, y), 0).x;
}

fn Py(tex: texture_2d<f32>, y: i32, x: i32) -> f32 {
  return textureLoad(tex, vec2<i32>(x, y), 0).y;
}

fn StoreR(dst: texture_storage_2d<r32float, write>, y: u32, x: u32, v: f32) {
  textureStore(dst, vec2<i32>(i32(x), i32(y)), vec4<f32>(v, 0.0, 0.0, 1.0));
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

fn LowPass(raw: texture_2d<f32>, y: i32, x: i32) -> f32 {
  return 0.25 * Px(raw, y, x) +
    0.125 * (Px(raw, y - 1, x) + Px(raw, y + 1, x) + Px(raw, y, x - 1) + Px(raw, y, x + 1)) +
    0.0625 * (Px(raw, y - 1, x - 1) + Px(raw, y - 1, x + 1) +
              Px(raw, y + 1, x - 1) + Px(raw, y + 1, x + 1));
}

// ---------------------------------------------------------------------------
// Kernel 1: rcd_init_and_vh
// ---------------------------------------------------------------------------
@group(0) @binding(0) var raw_init: texture_2d<f32>;
@group(0) @binding(1) var g_init: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var dir_init: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params_init: SinglePlaneParams;

@compute @workgroup_size(32, 8, 1)
fn rcd_init_and_vh(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_init.width || gid.y >= params_init.height) {
    return;
  }

  let val = Px(raw_init, i32(gid.y), i32(gid.x));
  let color = FC(params_init, gid.y, gid.x);

  StoreR(g_init, gid.y, gid.x, select(0.0, val, color == 1u));

  var vh = 0.0;
  var pq = 0.0;
  if (gid.x >= 4u && gid.y >= 4u && gid.x + 4u < params_init.width && gid.y + 4u < params_init.height) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = val;

    let vm1 = Px(raw_init, y - 1, x);
    let vp1 = Px(raw_init, y + 1, x);
    let vm2 = Px(raw_init, y - 2, x);
    let vp2 = Px(raw_init, y + 2, x);
    let vm3 = Px(raw_init, y - 3, x);
    let vp3 = Px(raw_init, y + 3, x);
    let vm4 = Px(raw_init, y - 4, x);
    let vp4 = Px(raw_init, y + 4, x);

    let hm1 = Px(raw_init, y, x - 1);
    let hp1 = Px(raw_init, y, x + 1);
    let hm2 = Px(raw_init, y, x - 2);
    let hp2 = Px(raw_init, y, x + 2);
    let hm3 = Px(raw_init, y, x - 3);
    let hp3 = Px(raw_init, y, x + 3);
    let hm4 = Px(raw_init, y, x - 4);
    let hp4 = Px(raw_init, y, x + 4);

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
      vm4 * vm4 + vp4 * vp4,
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
      hm4 * hm4 + hp4 * hp4,
      kEpsSq
    );

    vh = V_stat / (V_stat + H_stat);

    if (color != 1u) {
      let nw1 = Px(raw_init, y - 1, x - 1);
      let se1 = Px(raw_init, y + 1, x + 1);
      let nw2 = Px(raw_init, y - 2, x - 2);
      let se2 = Px(raw_init, y + 2, x + 2);
      let nw3 = Px(raw_init, y - 3, x - 3);
      let se3 = Px(raw_init, y + 3, x + 3);
      let nw4 = Px(raw_init, y - 4, x - 4);
      let se4 = Px(raw_init, y + 4, x + 4);
      let sw1 = Px(raw_init, y + 1, x - 1);
      let ne1 = Px(raw_init, y - 1, x + 1);
      let sw2 = Px(raw_init, y + 2, x - 2);
      let ne2 = Px(raw_init, y - 2, x + 2);
      let sw3 = Px(raw_init, y + 3, x - 3);
      let ne3 = Px(raw_init, y - 3, x + 3);
      let sw4 = Px(raw_init, y + 4, x - 4);
      let ne4 = Px(raw_init, y - 4, x + 4);
      let P_stat = max(-18.0*c*nw1 - 18.0*c*se1 - 36.0*c*nw2 - 36.0*c*se2 + 18.0*c*nw3 +
        18.0*c*se3 - 2.0*c*nw4 - 2.0*c*se4 + 38.0*c*c - 70.0*nw1*se1 -
        12.0*nw1*nw2 + 24.0*nw1*se2 - 38.0*nw1*nw3 + 16.0*nw1*se3 +
        12.0*nw1*nw4 - 6.0*nw1*se4 + 46.0*nw1*nw1 + 24.0*se1*nw2 -
        12.0*se1*se2 + 16.0*se1*nw3 - 38.0*se1*se3 - 6.0*se1*nw4 +
        12.0*se1*se4 + 46.0*se1*se1 + 14.0*nw2*se2 - 12.0*nw2*se3 -
        2.0*nw2*nw4 + 2.0*nw2*se4 + 11.0*nw2*nw2 - 12.0*se2*nw3 +
        2.0*se2*nw4 - 2.0*se2*se4 + 11.0*se2*se2 + 2.0*nw3*se3 -
        6.0*nw3*nw4 + 10.0*nw3*nw3 - 6.0*se3*se4 + 10.0*se3*se3 +
        nw4*nw4 + se4*se4, kEpsSq);
      let Q_stat = max(-18.0*c*sw1 - 18.0*c*ne1 - 36.0*c*sw2 - 36.0*c*ne2 + 18.0*c*sw3 +
        18.0*c*ne3 - 2.0*c*sw4 - 2.0*c*ne4 + 38.0*c*c - 70.0*sw1*ne1 -
        12.0*sw1*sw2 + 24.0*sw1*ne2 - 38.0*sw1*sw3 + 16.0*sw1*ne3 +
        12.0*sw1*sw4 - 6.0*sw1*ne4 + 46.0*sw1*sw1 + 24.0*ne1*sw2 -
        12.0*ne1*ne2 + 16.0*ne1*sw3 - 38.0*ne1*ne3 - 6.0*ne1*sw4 +
        12.0*ne1*ne4 + 46.0*ne1*ne1 + 14.0*sw2*ne2 - 12.0*sw2*ne3 -
        2.0*sw2*sw4 + 2.0*sw2*ne4 + 11.0*sw2*sw2 - 12.0*ne2*sw3 +
        2.0*ne2*sw4 - 2.0*ne2*ne4 + 11.0*ne2*ne2 + 2.0*sw3*ne3 -
        6.0*sw3*sw4 + 10.0*sw3*sw3 - 6.0*ne3*ne4 + 10.0*ne3*ne3 +
        sw4*sw4 + ne4*ne4, kEpsSq);
      pq = P_stat / (P_stat + Q_stat);
    }
  }

  textureStore(dir_init, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(vh, pq, 0.0, 1.0));
}

// ---------------------------------------------------------------------------
// Kernel 2: rcd_green_at_rb
// ---------------------------------------------------------------------------
@group(0) @binding(0) var raw_green: texture_2d<f32>;
@group(0) @binding(1) var vh_dir_green: texture_2d<f32>;
@group(0) @binding(2) var g_green_src: texture_2d<f32>;
@group(0) @binding(3) var g_green_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var<uniform> params_green: SinglePlaneParams;

@compute @workgroup_size(32, 8, 1)
fn rcd_green_at_rb(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_green.width || gid.y >= params_green.height) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  var out_g = Px(g_green_src, y, x);
  if (gid.x >= 4u && gid.y >= 4u && gid.x + 4u < params_green.width &&
      gid.y + 4u < params_green.height && FC(params_green, gid.y, gid.x) != 1u) {
    let VH_central = Px(vh_dir_green, y, x);
    let VH_neigh = 0.25 * (Px(vh_dir_green, y - 1, x - 1) + Px(vh_dir_green, y - 1, x + 1) +
                           Px(vh_dir_green, y + 1, x - 1) + Px(vh_dir_green, y + 1, x + 1));
    let VH_disc = select(VH_central, VH_neigh, abs(0.5 - VH_central) < abs(0.5 - VH_neigh));

    let c = Px(raw_green, y, x);
    let vm1 = Px(raw_green, y - 1, x);
    let vp1 = Px(raw_green, y + 1, x);
    let vm2 = Px(raw_green, y - 2, x);
    let vp2 = Px(raw_green, y + 2, x);
    let vm3 = Px(raw_green, y - 3, x);
    let vp3 = Px(raw_green, y + 3, x);
    let vm4 = Px(raw_green, y - 4, x);
    let vp4 = Px(raw_green, y + 4, x);
    let hm1 = Px(raw_green, y, x - 1);
    let hp1 = Px(raw_green, y, x + 1);
    let hm2 = Px(raw_green, y, x - 2);
    let hp2 = Px(raw_green, y, x + 2);
    let hm3 = Px(raw_green, y, x - 3);
    let hp3 = Px(raw_green, y, x + 3);
    let hm4 = Px(raw_green, y, x - 4);
    let hp4 = Px(raw_green, y, x + 4);

    let lpf_c = LowPass(raw_green, y, x);
    let lpf_n2 = LowPass(raw_green, y - 2, x);
    let lpf_s2 = LowPass(raw_green, y + 2, x);
    let lpf_w2 = LowPass(raw_green, y, x - 2);
    let lpf_e2 = LowPass(raw_green, y, x + 2);

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
    out_g = max(VH_disc * H_est + (1.0 - VH_disc) * V_est, 0.0);
  }
  StoreR(g_green_dst, gid.y, gid.x, out_g);
}

// ---------------------------------------------------------------------------
// Kernel 3: rcd_final_rgba
// ---------------------------------------------------------------------------
@group(0) @binding(0) var dir_final: texture_2d<f32>;
@group(0) @binding(1) var g_final: texture_2d<f32>;
@group(0) @binding(2) var raw_final: texture_2d<f32>;
@group(0) @binding(3) var out_final_rgba: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var<uniform> params_final: SinglePlaneParams;

fn RbChannelAfterRbAtRb(target_color: u32, y: i32, x: i32) -> f32 {
  let uy = u32(y);
  let ux = u32(x);
  let color = FC(params_final, uy, ux);
  let raw = Px(raw_final, y, x);
  if (color == target_color) {
    return raw;
  }
  if (color == 1u || ux < 4u || uy < 4u ||
      ux + 4u >= params_final.width || uy + 4u >= params_final.height) {
    return 0.0;
  }

  let PQ_c = Py(dir_final, y, x);
  let PQ_n = 0.25 * (Py(dir_final, y - 1, x - 1) + Py(dir_final, y - 1, x + 1) +
                     Py(dir_final, y + 1, x - 1) + Py(dir_final, y + 1, x + 1));
  let PQ_disc = select(PQ_c, PQ_n, abs(0.5 - PQ_c) < abs(0.5 - PQ_n));
  let g_c = Px(g_final, y, x);
  let ch_nw1 = Px(raw_final, y - 1, x - 1);
  let ch_ne1 = Px(raw_final, y - 1, x + 1);
  let ch_sw1 = Px(raw_final, y + 1, x - 1);
  let ch_se1 = Px(raw_final, y + 1, x + 1);
  let ch_nw3 = Px(raw_final, y - 3, x - 3);
  let ch_ne3 = Px(raw_final, y - 3, x + 3);
  let ch_sw3 = Px(raw_final, y + 3, x - 3);
  let ch_se3 = Px(raw_final, y + 3, x + 3);
  let g_nw2 = Px(g_final, y - 2, x - 2);
  let g_ne2 = Px(g_final, y - 2, x + 2);
  let g_sw2 = Px(g_final, y + 2, x - 2);
  let g_se2 = Px(g_final, y + 2, x + 2);
  let NW_grad = kEps + abs(ch_nw1 - ch_se1) + abs(ch_nw1 - ch_nw3) + abs(g_c - g_nw2);
  let NE_grad = kEps + abs(ch_ne1 - ch_sw1) + abs(ch_ne1 - ch_ne3) + abs(g_c - g_ne2);
  let SW_grad = kEps + abs(ch_sw1 - ch_ne1) + abs(ch_sw1 - ch_sw3) + abs(g_c - g_sw2);
  let SE_grad = kEps + abs(ch_se1 - ch_nw1) + abs(ch_se1 - ch_se3) + abs(g_c - g_se2);
  let NW_est = ch_nw1 - Px(g_final, y - 1, x - 1);
  let NE_est = ch_ne1 - Px(g_final, y - 1, x + 1);
  let SW_est = ch_sw1 - Px(g_final, y + 1, x - 1);
  let SE_est = ch_se1 - Px(g_final, y + 1, x + 1);
  let P_est = (NW_grad * SE_est + SE_grad * NW_est) / (NW_grad + SE_grad);
  let Q_est = (NE_grad * SW_est + SW_grad * NE_est) / (NE_grad + SW_grad);
  return max(0.0, g_c + (1.0 - PQ_disc) * P_est + PQ_disc * Q_est);
}

fn ReconstructTargetAtGreen(target_color: u32, y: i32, x: i32, vh_disc: f32) -> f32 {
  return ReconstructRbAtGreen(
    RbChannelAfterRbAtRb(target_color, y - 1, x),
    RbChannelAfterRbAtRb(target_color, y + 1, x),
    RbChannelAfterRbAtRb(target_color, y - 3, x),
    RbChannelAfterRbAtRb(target_color, y + 3, x),
    RbChannelAfterRbAtRb(target_color, y, x - 1),
    RbChannelAfterRbAtRb(target_color, y, x + 1),
    RbChannelAfterRbAtRb(target_color, y, x - 3),
    RbChannelAfterRbAtRb(target_color, y, x + 3),
    Px(g_final, y, x),
    Px(g_final, y - 2, x),
    Px(g_final, y + 2, x),
    Px(g_final, y, x - 2),
    Px(g_final, y, x + 2),
    Px(g_final, y - 1, x),
    Px(g_final, y + 1, x),
    Px(g_final, y, x - 1),
    Px(g_final, y, x + 1),
    vh_disc);
}

@compute @workgroup_size(32, 8, 1)
fn rcd_final_rgba(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params_final.width || gid.y >= params_final.height) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let color = FC(params_final, gid.y, gid.x);
  let raw = Px(raw_final, y, x);
  var out_r = select(0.0, raw, color == 0u);
  let out_g = Px(g_final, y, x);
  var out_b = select(0.0, raw, color == 2u);

  if (color == 0u) {
    out_b = RbChannelAfterRbAtRb(2u, y, x);
  } else if (color == 2u) {
    out_r = RbChannelAfterRbAtRb(0u, y, x);
  } else if (gid.x >= 4u && gid.y >= 4u && gid.x + 4u < params_final.width &&
             gid.y + 4u < params_final.height) {
    let VH_central = Px(dir_final, y, x);
    let VH_neigh = 0.25 * (Px(dir_final, y - 1, x - 1) + Px(dir_final, y - 1, x + 1) +
                           Px(dir_final, y + 1, x - 1) + Px(dir_final, y + 1, x + 1));
    let VH_disc = select(VH_central, VH_neigh, abs(0.5 - VH_central) < abs(0.5 - VH_neigh));
    out_r = ReconstructTargetAtGreen(0u, y, x, VH_disc);
    out_b = ReconstructTargetAtGreen(2u, y, x, VH_disc);
  }

  textureStore(out_final_rgba, vec2<i32>(x, y), vec4<f32>(out_r, out_g, out_b, 1.0));
}
