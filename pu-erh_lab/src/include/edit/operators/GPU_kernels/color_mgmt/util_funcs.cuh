//  Copyright 2025 Yurun Zi
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

// CUDA implementations of helper functions in Lib.Academy.OutputTransform.ctl
// Reference:
// https://github.com/aces-aswf/aces-core/blob/dev/lib/Lib.Academy.OutputTransform.ctl
#pragma once

#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>

#include "edit/operators/GPU_kernels/param.cuh"
#include "util_const.cuh"

namespace puerhlab {
namespace CUDA {
#define REF_LUM 100.0f

__device__ __forceinline__ float3 Mult_f4_f33(const float4& v, const float m[9]) {
  return make_float3(fmaf(m[0], v.x, fmaf(m[1], v.y, m[2] * v.z)),
                     fmaf(m[3], v.x, fmaf(m[4], v.y, m[5] * v.z)),
                     fmaf(m[6], v.x, fmaf(m[7], v.y, m[8] * v.z)));
}

__device__ __forceinline__ float3 Mult_f3_f33(const float3& v, const float m[9]) {
  return make_float3(fmaf(m[0], v.x, fmaf(m[1], v.y, m[2] * v.z)),
                     fmaf(m[3], v.x, fmaf(m[4], v.y, m[5] * v.z)),
                     fmaf(m[6], v.x, fmaf(m[7], v.y, m[8] * v.z)));
}

__device__ __forceinline__ float3 Mult_f_f3(const float& s, const float3& v) {
  return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ __forceinline__ float3 clampf3(const float3& v, float min_val, float max_val) {
  return make_float3(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val),
                     fminf(fmaxf(v.z, min_val), max_val));
}

__device__ __forceinline__ int   sign(float v) { return (v > 0) - (v < 0); }

__device__ __forceinline__ float signf(float v) { return (v > 0.f) - (v < 0.f); }

__device__ __forceinline__ float spow(float base, float exponent) {
  float ei     = nearbyintf(exponent);
  int   is_int = (fabsf(exponent - ei) < 1e-6f);

  float p      = powf(fabsf(base), exponent);

  int   e_int  = (int)ei;
  int   odd    = (e_int & 1);
  float sgn    = 1.0f - 2.0f * (float)((base < 0.0f) & is_int & odd);
  p *= sgn;

  float ok = (float)((base >= 0.0f) | is_int);
  return p * ok;
}

__device__ __forceinline__ float wrap360f(float hue) {
  float y = fmodf(hue, 360.0f);
  y += 360.0f * (float)(y < 0.0f);
  return y;
}

__device__ __forceinline__ float3 pa_nlrc_fwd(const float3& RGB, float F_L) {
  float3 F_L_RGB = make_float3(spow((F_L * fabs(RGB.x)) / 100.f, 0.42f),
                               spow((F_L * fabs(RGB.y)) / 100.f, 0.42f),
                               spow((F_L * fabs(RGB.z)) / 100.f, 0.42f));

  return make_float3(400.f * copysignf(1.f, RGB.x) * F_L_RGB.x / (27.13f + F_L_RGB.x),
                     400.f * copysignf(1.f, RGB.y) * F_L_RGB.y / (27.13f + F_L_RGB.y),
                     400.f * copysignf(1.f, RGB.z) * F_L_RGB.z / (27.13f + F_L_RGB.z));
}

// post_adaptation_non_linear_response_compression_inverse
__device__ __forceinline__ float3 pa_nlrc_inv(const float3& RGB, float F_L) {
  return make_float3(sign(RGB.x) * 100.f / F_L *
                         spow((27.13f * fabs(RGB.x)) / (400.f - fabs(RGB.x)), 1.0f / 0.42f),
                     sign(RGB.y) * 100.f / F_L *
                         spow((27.13f * fabs(RGB.y)) / (400.f - fabs(RGB.y)), 1.0f / 0.42f),
                     sign(RGB.z) * 100.f / F_L *
                         spow((27.13f * fabs(RGB.z)) / (400.f - fabs(RGB.z)), 1.0f / 0.42f));
}

__device__ __forceinline__ float3 XYZ_to_JMh(const float3& XYZ, const float3& XYZ_w,
                                             GPU_ODTParams* odt_param) {
  float  Y_w    = XYZ_w.y;

  float3 RGB_w  = Mult_f3_f33(XYZ_w, CAM16_XYZ_TO_RGB);
  float  D      = 1.f;

  float  n      = Y_b_CONST / Y_w;
  float  z      = 1.48f + sqrtf(n);

  float3 D_RGB  = make_float3(D * (Y_w / RGB_w.x) + 1.f - D, D * (Y_w / RGB_w.y) + 1.f - D,
                              D * (Y_w / RGB_w.z) + 1.f - D);
  float3 RGB_wc = make_float3(D_RGB.x * RGB_w.x, D_RGB.y * RGB_w.y, D_RGB.z * RGB_w.z);

  float3 RGB_aw = pa_nlrc_fwd(RGB_wc, F_L_CONST);
  float  A_w    = ra * RGB_aw.x + RGB_aw.y + ba * RGB_aw.z;

  // Step 1 - Converting CIE XYZ tristimulus values to sharpened RGB values
  float3 RGB    = Mult_f3_f33(XYZ, CAM16_XYZ_TO_RGB);

  // Step 2
  float3 RGB_c  = make_float3(D_RGB.x * RGB.x, D_RGB.y * RGB.y, D_RGB.z * RGB.z);

  // Step 3
  float3 RGB_a  = pa_nlrc_fwd(RGB_c, F_L_CONST);

  // Step 4 - Converting to preliminary cartesian coordinates
  float  a      = RGB_a.x - 12.f * RGB_a.y / 11.f + RGB_a.z / 11.f;
  float  b      = (RGB_a.x + RGB_a.y - 2.f * RGB_a.z) / 9.f;

  // Computing the hue angle
  float  h_rad  = atan2f(b, a);
  float  h_deg  = wrap360f(h_rad * 180.0f / 3.14159265f);

  // Step 6 - Computing achromatic responses for the stimulus
  float  A      = ra * RGB_a.x + RGB_a.y + ba * RGB_a.z;

  // Step 7 - Computing the correlate of lightness, J
  float  J      = 100.f * pow(A / A_w, surround[1] * z);

  // Step 8 - Computing the correlate of colourfulness, M
  float  M      = (43.f * surround[2] * sqrtf(a * a + b * b)) * (float)(J > 0.0f);

  return make_float3(J, M, h_deg);
}

float3 __device__ __forceinline__ JMh_to_XYZ(const float3& JMh, const float3& XYZ_w,
                                             GPU_ODTParams* odt_param) {
  // To be implemented if needed
  float  J      = JMh.x;
  float  M      = JMh.y;
  float  h      = JMh.z;

  float  Y_w    = XYZ_w.y;

  float3 RGB_w  = Mult_f3_f33(XYZ_w, CAM16_XYZ_TO_RGB);

  float  D      = 1.0f;
  float  n      = Y_b_CONST / Y_w;
  float  z      = 1.48f + sqrtf(n);

  float3 D_RGB  = make_float3(D * (Y_w / RGB_w.x) + 1.f - D, D * (Y_w / RGB_w.y) + 1.f - D,
                              D * (Y_w / RGB_w.z) + 1.f - D);

  float3 RGB_wc = make_float3(D_RGB.x * RGB_w.x, D_RGB.y * RGB_w.y, D_RGB.z * RGB_w.z);

  float3 RGB_aw = pa_nlrc_fwd(RGB_wc, F_L_CONST);

  float  A_w    = ra * RGB_aw.x + RGB_aw.y + ba * RGB_aw.z;

  float  hr     = h * 3.14159265f / 180.0f;
  float  A      = A_w * spow(J / 100.f, 1.f / (surround[1] * z));

  // Computing achromatic respons A for the stimulus
  float  P_p_1  = 43.f * surround[2];
  float  P_p_2  = A;

  // Step 3 - Computing opponent colour dimensions a and b
  float  gamma  = M / P_p_1;
  float  a      = gamma * cosf(hr);
  float  b      = gamma * sinf(hr);

  // Step 4 - Applying post-adaptation non-linear response compression matrix
  float3 vec    = make_float3(P_p_2, a, b);
  float3 RGB_a  = Mult_f_f3(1.f / 1403.f, Mult_f3_f33(vec, panlrcm));

  // Step 5 - Inverting the post-adaptation non-linear response compression
  float3 RGB_c  = pa_nlrc_inv(RGB_a, F_L_CONST);

  // Step 6
  float3 RGB    = make_float3(RGB_c.x / D_RGB.x, RGB_c.y / D_RGB.y, RGB_c.z / D_RGB.z);

  // Step 7 - Converting sharpened RGB to CIE XYZ tristimulus values
  return Mult_f3_f33(RGB, CAM16_XYZ_TO_RGB);
}

// Not exactly the same as the original ACES implementation
// The input will be directly in AP1 space
__device__ __forceinline__ float3 ACES_To_JMh(const float4& aces, float peak_lum,
                                              GPU_ODTParams* odt_param) {
  // AP0 to XYZ
  float3 ap1_clamped = clampf3(make_float3(aces.x, aces.y, aces.z), 0.0f, odt_param->upper_clamp_);
  float3 XYZ         = Mult_f3_f33(ap1_clamped, AP1_to_XYZ);

  float3 RGB_w   = make_float3(REF_LUM, REF_LUM, REF_LUM);  // Reference luminance white point D65
  float3 XYZ_w   = Mult_f3_f33(RGB_w, AP0_to_XYZ);

  float3 XYZ_lum = Mult_f_f3(REF_LUM, XYZ);

  return XYZ_to_JMh(XYZ_lum, XYZ_w, odt_param);
}

__device__ __forceinline__ float3 JMh_To_ACES(const float3& JMh, float peak_lum,
                                              GPU_ODTParams* odt_param) {
  const float3 RGB_w =
      make_float3(REF_LUM, REF_LUM, REF_LUM);  // Reference luminance white point D65
  float3 XYZ_w_ACES = Mult_f3_f33(RGB_w, AP0_to_XYZ);

  // JMh to XYZ
  float3 XYZ_lum    = JMh_to_XYZ(JMh, XYZ_w_ACES, odt_param);
  float3 XYZ        = Mult_f_f3(1.f / REF_LUM, XYZ_lum);

  // XYZ to AP0
  return Mult_f3_f33(XYZ, AP0_XYZ_TO_RGB);
}

__device__ __forceinline__ float Hellwig_J_to_Y(float J, float surround = 0.59f, float L_A = 100.f,
                                                float Y_b = 20.f) {
  float A = A_W_CONST * powf(fabs(J) / 100.f, 1.f / (surround * z_CONST));
  return signf(J) * 100.f / F_L_CONST * powf((27.13f * A) / (400.f - A), 1.f / 0.42f);
}

__device__ __forceinline__ float Hellwig_Y_to_J(float Y, float surround = 0.59f, float L_A = 100.f,
                                                float Y_b = 20.f) {
  float F_L_Y = powf((F_L_CONST * fabs(Y)) / 100.f, 0.42f);
  return signf(Y) * 100.f *
         powf((400.f * F_L_Y) / (F_L_Y + 27.13f) / A_W_CONST, surround * z_CONST);
}

__device__ __forceinline__ float Tonescale_fwd(float x, GPU_TSParams* params) {
  float f = params->m_2_ * powf(fmaxf(0.0, x) / (x + params->s_2_), params->g_);
  float h = fmaxf(0., f * f / (f + params->t_1_));

  return h * params->n_r_;
}

__device__ __forceinline__ float2 CuspFromTable(float hue, float* TABLE) {
  float3 lo;
  float3 hi;
}

__device__ __forceinline__ float  ReachFromTable(float hue, float* TABLE) {
  // To be implemented if needed
  return 0.f;
}

// A "toe" function that remaps the given value x between 0 and limit.
// The k1 and k2 parameters change the size and shape of the toe.
// https://www.desmos.com/calculator/6vplvw14ti
__device__ __forceinline__ float toe(float x, float limit, float k1_in, float k2_in,
                                     bool invert = false) {
  // Clamp k2 to avoid division by 0 / extreme slope
  float k2      = fmaxf(k2_in, 1e-3f);
  float k1      = sqrtf(k1_in * k1_in + k2 * k2);
  float k3      = (limit + k1) / (limit + k2);

  // forward branch
  float t       = k3 * x - k1;
  float fwd     = 0.5f * (t + sqrtf(t * t + 4.0f * k2 * k3 * x));

  // inverse branch
  float inv     = (x * x + k1 * x) / (k3 * (x + k2));

  // select invert (invert must be 0/1)
  float invMask = (float)invert;                    // 0.0 or 1.0
  float toeVal  = fmaf(invMask, (inv - fwd), fwd);  // toeVal = invert ? inv : fwd

  // select x>limit
  // (x>limit) is still a compare, but no control-flow branch is required.
  float gtMask  = (x > limit) ? 1.0f : 0.0f;  // may compile to predicate
  return fmaf(gtMask, (x - toeVal), toeVal);  // gt? x : toeVal
}

__device__ __forceinline__ float ChromaCompression_fwd(float3& output_JMh, float origJ,
                                                       GPU_ODTParams* params,
                                                       float*         REACH_GAMUT_TABLE,
                                                       float*         REACH_CUSP_TABLE) {
  // To be implemented if needed
  float J = output_JMh.x;
  float M = output_JMh.y;
  float h = output_JMh.z;

  if (M == 0.f) return M;

  float  nJ    = J / params->limit_J_max_;
  float  snJ   = fmaxf(0.f, 1.f - nJ);
  float2 cusp  = CuspFromTable(h, REACH_CUSP_TABLE);
  float  M_nor = cusp.x;

  float  limit = powf(nJ, params->model_gamma_) * ReachFromTable(h, REACH_CUSP_TABLE) / M_nor;

  // Rescaling of M with the tonescaled J to get the M to the same range as
  // J after the tonescale.  The rescaling uses the Hellwig2022 model gamma to
  // keep the M/J ratio correct (keeping the chromaticities constant).
  M            = M * powf(J / origJ, params->model_gamma_);

  // Normalize M with the rendering space cusp M
  M            = M / M_nor;

  // Expand the colorfulness by running the toe function in reverse.  The goal is to
  // expand less saturated colors less and more saturated colors more.  The expansion
  // increases saturation in the shadows and mid-tones but not in the highlights.
  // The 0.001 offset starts the expansions slightly above zero.  The sat_thr makes
  // the toe less aggressive near black to reduce the expansion of noise.
  M            = limit -
      toe(limit - M, limit - 0.001f, snJ * params->sat_, sqrtf(nJ * nJ + params->sat_thr_), false);

  // Compress the colorfulness.  The goal is to compress less saturated colors more and
  // more saturated colors less, especially in the highlights.  This step creates the
  // saturation roll-off in the highlights, but attemps to preserve pure colors.  This
  // mostly affects highlights and mid-tones, and does not compress shadows.
  M = toe(M, limit, nJ * params->compr_, snJ, false);

  // Denormalize M
  return M * M_nor;
}

__device__ __forceinline__ float3 ToneMapAndCompress_fwd(float3&        input_JMh,
                                                         GPU_ODTParams* odt_param,
                                                         float*         REACH_GAMUT_TABLE,
                                                         float*         REACH_CUSP_TABLE) {
  float  linear        = Hellwig_J_to_Y(input_JMh.x) / REF_LUM;
  float  luminance_TS  = Tonescale_fwd(linear, &odt_param->ts_params_);
  float  tone_mapped_J = Hellwig_Y_to_J(luminance_TS * REF_LUM);

  float3 output_JMh    = make_float3(tone_mapped_J, input_JMh.y, input_JMh.z);
  output_JMh.y = ChromaCompression_fwd(output_JMh, input_JMh.x, odt_param, REACH_GAMUT_TABLE,
                                       REACH_CUSP_TABLE);
  return output_JMh;
}
};  // namespace CUDA
};  // namespace puerhlab