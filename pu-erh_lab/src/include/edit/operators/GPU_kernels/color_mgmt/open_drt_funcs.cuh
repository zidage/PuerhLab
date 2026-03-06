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

#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "edit/operators/GPU_kernels/color_mgmt/util_funcs.cuh"
#include "edit/operators/GPU_kernels/param.cuh"

namespace puerhlab {
namespace CUDA {

GPU_FUNC float odrt_clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

GPU_FUNC float odrt_smoothstep(float edge0, float edge1, float x) {
  const float width = fmaxf(edge1 - edge0, 1e-6f);
  const float t     = odrt_clamp((x - edge0) / width, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

GPU_FUNC float odrt_wrap_degrees(float hue) {
  float wrapped = fmodf(hue, 360.0f);
  if (wrapped < 0.0f) {
    wrapped += 360.0f;
  }
  return wrapped;
}

GPU_FUNC float odrt_hue_distance(float a, float b) {
  const float delta = fabsf(odrt_wrap_degrees(a - b));
  return fminf(delta, 360.0f - delta);
}

GPU_FUNC float odrt_sector_weight(float hue, float center, float range_scale) {
  const float half_width = fmaxf(12.0f, 30.0f * fmaxf(range_scale, 0.05f));
  const float dist       = odrt_hue_distance(hue, center);
  return odrt_clamp(1.0f - dist / half_width, 0.0f, 1.0f);
}

GPU_FUNC float odrt_luma(const float3& rgb, const float weights[3]) {
  return rgb.x * weights[0] + rgb.y * weights[1] + rgb.z * weights[2];
}

GPU_FUNC float3 odrt_lerp_f3(const float3& a, const float3& b, float t) {
  return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

GPU_FUNC float3 odrt_limit_preserve_chroma(float3 rgb, float lower, float upper) {
  if (!isfinite(rgb.x) || !isfinite(rgb.y) || !isfinite(rgb.z)) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  rgb.x = fmaxf(rgb.x, lower);
  rgb.y = fmaxf(rgb.y, lower);
  rgb.z = fmaxf(rgb.z, lower);

  const float max_channel = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
  if (max_channel > upper && max_channel > 1e-6f) {
    const float scale = upper / max_channel;
    rgb               = mult_f_f3(rgb, scale);
  }
  return rgb;
}

GPU_FUNC float odrt_compress_toe_quadratic(float x, float toe, bool inverse = false) {
  if (toe <= 0.0f) {
    return x;
  }

  if (!inverse) {
    const float abs_x = fabsf(x);
    return copysignf((x * x) / (abs_x + toe), x);
  }

  const float abs_x = fabsf(x);
  const float value = 0.5f * (abs_x + sqrtf(abs_x * (4.0f * toe + abs_x)));
  return copysignf(value, x);
}

GPU_FUNC float odrt_hue_from_rgb(const float3& rgb) {
  const float maxc  = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
  const float minc  = fminf(rgb.x, fminf(rgb.y, rgb.z));
  const float delta = maxc - minc;
  if (delta <= 1e-6f) {
    return 0.0f;
  }

  float hue = 0.0f;
  if (maxc == rgb.x) {
    hue = 60.0f * ((rgb.y - rgb.z) / delta);
    if (rgb.y < rgb.z) {
      hue += 360.0f;
    }
  } else if (maxc == rgb.y) {
    hue = 60.0f * (((rgb.z - rgb.x) / delta) + 2.0f);
  } else {
    hue = 60.0f * (((rgb.x - rgb.y) / delta) + 4.0f);
  }

  return odrt_wrap_degrees(hue);
}

GPU_FUNC float odrt_saturation_from_rgb(const float3& rgb) {
  const float maxc = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
  const float minc = fminf(rgb.x, fminf(rgb.y, rgb.z));
  if (maxc <= 1e-6f) {
    return 0.0f;
  }
  return (maxc - minc) / maxc;
}

GPU_FUNC float odrt_shape_scene_luminance(float y, const GPU_OpenDRTParams& p) {
  float shaped = fmaxf(y, 0.0f);

  const float low_width  = fmaxf(0.05f, 0.18f * fmaxf(p.tn_lcon_w_, 0.2f));
  const float low_mask   = 1.0f - odrt_smoothstep(0.0f, low_width, shaped);
  if (p.tn_lcon_enable_) {
    const float lift = 1.0f + 0.35f * fmaxf(p.tn_lcon_, 0.0f) * low_mask;
    shaped           = powf(fmaxf(shaped, 0.0f), 1.0f / fmaxf(lift, 0.1f));
  }

  const float high_pivot = fmaxf(0.18f, p.tn_hcon_pv_);
  const float high_width = fmaxf(0.1f, p.tn_hcon_st_);
  const float high_mask  = odrt_smoothstep(high_pivot, high_pivot + high_width, shaped);
  if (p.tn_hcon_enable_) {
    const float shoulder = 1.0f + 0.25f * fmaxf(p.tn_hcon_, 0.0f) * high_mask;
    shaped               = powf(fmaxf(shaped, 0.0f), shoulder);
  }

  return shaped;
}

GPU_FUNC float odrt_tonescale_fwd(float scene_y, const GPU_OpenDRTParams& p) {
  const float shaped = odrt_shape_scene_luminance(scene_y, p);
  const float denom  = shaped + fmaxf(p.ts_s_, 1e-6f);
  const float ratio  = (denom > 1e-6f) ? (shaped / denom) : 0.0f;
  const float mm     = p.ts_m2_ * powf(fmaxf(ratio, 0.0f), fmaxf(p.ts_p_, 0.01f));
  return odrt_compress_toe_quadratic(mm, fmaxf(p.tn_toe_, 0.0f), false);
}

GPU_FUNC float odrt_window(float x, float center, float radius, float sharpness) {
  const float dist = fabsf(x - center) / fmaxf(radius, 1e-4f);
  return odrt_clamp(1.0f - dist * fmaxf(sharpness, 0.0f), 0.0f, 1.0f);
}

GPU_FUNC float3 OpenDRTOutputTransform_fwd(const float3& in_color, const GPU_OpenDRTParams& p) {
  if (!isfinite(in_color.x) || !isfinite(in_color.y) || !isfinite(in_color.z)) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  float3 render_rgb = mult_f3_f33(in_color, p.ap1_to_render_mat_);
  render_rgb        = odrt_limit_preserve_chroma(render_rgb, 0.0f, 65504.0f);

  const float scene_y  = fmaxf(odrt_luma(render_rgb, p.rs_w_), 0.0f);
  const float display_y = odrt_tonescale_fwd(scene_y, p);
  const float y_scale   = (scene_y > 1e-6f) ? (display_y / scene_y) : 0.0f;
  float3      toned_rgb = mult_f_f3(render_rgb, y_scale);

  const float peak_norm = fmaxf(p.peak_luminance_ / 100.0f, 1e-4f);
  const float y_frac    = odrt_clamp(display_y / peak_norm, 0.0f, 4.0f);

  const float shadow_mask = 1.0f - odrt_smoothstep(0.03f, 0.35f, y_frac);
  const float mid_mask    =
      odrt_smoothstep(0.05f, 0.35f, y_frac) * (1.0f - odrt_smoothstep(0.45f, 0.85f, y_frac));
  const float high_mask   = odrt_smoothstep(0.45f, 0.85f, y_frac);
  const float peak_mask   = odrt_smoothstep(0.75f, 1.05f, y_frac);

  const float hue         = odrt_hue_from_rgb(toned_rgb);
  const float sat         = odrt_saturation_from_rgb(toned_rgb);

  const float wr          = odrt_sector_weight(hue, 0.0f, p.hs_r_rng_);
  const float wy          = odrt_sector_weight(hue, 60.0f, p.hs_y_rng_);
  const float wg          = odrt_sector_weight(hue, 120.0f, p.hs_g_rng_);
  const float wc          = odrt_sector_weight(hue, 180.0f, p.hs_c_rng_);
  const float wb          = odrt_sector_weight(hue, 240.0f, p.hs_b_rng_);
  const float wm          = odrt_sector_weight(hue, 300.0f, p.hs_m_rng_);

  float purity_gain       = 0.0f;
  if (p.pt_enable_) {
    purity_gain += shadow_mask * (p.pt_lml_ + wr * p.pt_lml_r_ + wg * p.pt_lml_g_ + wb * p.pt_lml_b_);
    purity_gain += high_mask * (p.pt_lmh_ + wr * p.pt_lmh_r_ + wb * p.pt_lmh_b_);
  }
  if (p.ptl_enable_) {
    purity_gain += shadow_mask * (wc * p.ptl_c_ + wm * p.ptl_m_ + wy * p.ptl_y_);
  }
  if (p.ptm_enable_) {
    purity_gain += odrt_window(y_frac, 0.25f, p.ptm_low_rng_, p.ptm_low_st_ + 1.0f) * p.ptm_low_;
    purity_gain += odrt_window(y_frac, 0.85f, p.ptm_high_rng_, p.ptm_high_st_ + 1.0f) * p.ptm_high_;
  }
  purity_gain *= (0.35f + 0.65f * p.pt_cmp_Lf_);

  const float highlight_rolloff =
      1.0f - odrt_smoothstep(0.55f, 1.10f + fmaxf(p.ts_s1_, 0.0f), y_frac);
  const float sat_scale =
      fmaxf(0.0f, 1.0f + 0.15f * p.rs_sa_ + 0.30f * purity_gain * highlight_rolloff);

  float3 chroma = make_float3(toned_rgb.x - display_y, toned_rgb.y - display_y, toned_rgb.z - display_y);
  chroma        = mult_f_f3(chroma, sat_scale);

  float3 bias   = make_float3(0.0f, 0.0f, 0.0f);
  if (p.hs_rgb_enable_) {
    bias.x += wr * p.hs_r_;
    bias.y += wg * p.hs_g_;
    bias.z += wb * p.hs_b_;
  }
  if (p.hs_cmy_enable_) {
    bias.x += 0.5f * (wm * p.hs_m_ + wy * p.hs_y_) - wc * p.hs_c_;
    bias.y += 0.5f * (wc * p.hs_c_ + wy * p.hs_y_) - wm * p.hs_m_;
    bias.z += 0.5f * (wc * p.hs_c_ + wm * p.hs_m_) - wy * p.hs_y_;
  }
  if (p.brl_enable_) {
    const float shadow_scale = shadow_mask * p.brl_st_ * (1.0f + p.brl_rng_);
    bias.x += shadow_scale * (p.brl_ + wr * p.brl_r_);
    bias.y += shadow_scale * (p.brl_ + wg * p.brl_g_);
    bias.z += shadow_scale * (p.brl_ + wb * p.brl_b_);
  }
  if (p.brlp_enable_) {
    bias.x += peak_mask * (p.brlp_ + wr * p.brlp_r_);
    bias.y += peak_mask * (p.brlp_ + wg * p.brlp_g_);
    bias.z += peak_mask * (p.brlp_ + wb * p.brlp_b_);
  }
  if (p.hc_enable_) {
    const float hc = high_mask * wr * p.hc_r_ * (1.0f + p.hc_r_rng_);
    bias.x += hc;
    bias.y -= 0.5f * hc;
    bias.z -= 0.5f * hc;
  }

  const float weighted_mean = odrt_luma(bias, p.rs_w_);
  bias.x -= weighted_mean;
  bias.y -= weighted_mean;
  bias.z -= weighted_mean;

  const float bias_scale = 0.08f * (0.25f + sat) * (shadow_mask + mid_mask + high_mask + peak_mask);
  float3      shaped_rgb = make_float3(display_y + chroma.x + bias_scale * bias.x,
                                  display_y + chroma.y + bias_scale * bias.y,
                                  display_y + chroma.z + bias_scale * bias.z);

  const float shaped_y   = odrt_luma(shaped_rgb, p.rs_w_);
  if (shaped_y > 1e-6f) {
    shaped_rgb = mult_f_f3(shaped_rgb, display_y / shaped_y);
  }

  const float creative_mix =
      odrt_clamp(p.creative_white_luminance_mix_ * odrt_smoothstep(0.18f, 1.0f, y_frac), 0.0f, 1.0f);
  const float3 limit_neutral = mult_f3_f33(shaped_rgb, p.render_to_limit_neutral_mat_);
  float3       limit_creative = mult_f3_f33(shaped_rgb, p.render_to_limit_creative_mat_);
  limit_creative              = mult_f_f3(limit_creative, p.creative_white_norm_);

  float3 limit_rgb            = odrt_lerp_f3(limit_neutral, limit_creative, creative_mix);
  limit_rgb                   = odrt_limit_preserve_chroma(limit_rgb, 0.0f, peak_norm);
  return limit_rgb;
}

}  // namespace CUDA
}  // namespace puerhlab
