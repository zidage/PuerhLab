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

#pragma once

#include <array>
#include <cmath>
#include <opencv2/core/matx.hpp>

namespace puerhlab {
namespace ColorUtils {

enum class ColorSpace : int {
  AP0,
  AP1,
  REC709,
  REC2020,
  P3_D65,
  PROPHOTO,
  ADOBE_RGB,
};

enum class ETOF : int {
  LINEAR,
  ST2084,
  HLG,
  GAMMA_2_6,
  BT1886,
  GAMMA_2_2,
  GAMMA_1_8,
};

struct ColorSpacePrimaries {
  float red_[2];
  float green_[2];
  float blue_[2];
  float white_[2];
};

const ColorSpacePrimaries AP0_PRIMARY = {
    {0.73470f, 0.26530f}, {0.00000f, 1.00000f}, {0.00010f, -0.07700f}, {0.32168f, 0.33767f}};

const ColorSpacePrimaries AP1_PRIMARY = {
    {0.713f, 0.293f}, {0.165f, 0.830f}, {0.128f, 0.044f}, {0.32168f, 0.33767f}};

const ColorSpacePrimaries REACH_PRIMARY = {
    {0.713f, 0.293f}, {0.165f, 0.830f}, {0.128f, 0.044f}, {0.32168f, 0.33767f}};

const ColorSpacePrimaries REC709_PRIMARY = {
    {0.640f, 0.330f}, {0.300f, 0.600f}, {0.150f, 0.060f}, {0.3127f, 0.3290f}};

const ColorSpacePrimaries REC2020_PRIMARY = {
    {0.708f, 0.292f}, {0.170f, 0.797f}, {0.131f, 0.046f}, {0.3127f, 0.3290f}};

const ColorSpacePrimaries P3_D65_PRIMARY = {
    {0.680f, 0.320f}, {0.265f, 0.690f}, {0.150f, 0.060f}, {0.3127f, 0.3290f}};

const ColorSpacePrimaries PROPHOTO_PRIMARY = {
    {0.734699f, 0.265301f}, {0.159597f, 0.840403f}, {0.036598f, 0.000105f}, {0.345704f, 0.358540f}};

const ColorSpacePrimaries ADOBE_RGB_PRIMARY = {
    {0.6400f, 0.3300f}, {0.2100f, 0.7100f}, {0.1500f, 0.0600f}, {0.3127f, 0.3290f}};

inline ColorSpacePrimaries SpaceEnumToPrimary(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::AP0:
      return AP0_PRIMARY;
    case ColorSpace::AP1:
      return AP1_PRIMARY;
    case ColorSpace::REC709:
      return REC709_PRIMARY;
    case ColorSpace::REC2020:
      return REC2020_PRIMARY;
    case ColorSpace::P3_D65:
      return P3_D65_PRIMARY;
    case ColorSpace::PROPHOTO:
      return PROPHOTO_PRIMARY;
    case ColorSpace::ADOBE_RGB:
      return ADOBE_RGB_PRIMARY;
    default:
      return REC709_PRIMARY;
  }
}

inline cv::Matx33f RGB_TO_XYZ_f33(const ColorSpacePrimaries& C, float Y = 1.0f) {
  // X and Z values of RGB value (1, 1, 1), or "white"
  float X = C.white_[0] * Y / C.white_[1];
  float Z = (1.f - C.white_[0] - C.white_[1]) * Y / C.white_[1];

  // Scale factors for matrix rows
  float d = C.red_[0] * (C.blue_[1] - C.green_[1]) + C.blue_[0] * (C.green_[1] - C.red_[1]) +
            C.green_[0] * (C.red_[1] - C.blue_[1]);

  float Sr = (X * (C.blue_[1] - C.green_[1]) -
              C.green_[0] * (Y * (C.blue_[1] - 1) + C.blue_[1] * (X + Z)) +
              C.blue_[0] * (Y * (C.green_[1] - 1) + C.green_[1] * (X + Z))) /
             d;

  float Sg =
      (X * (C.red_[1] - C.blue_[1]) + C.red_[0] * (Y * (C.blue_[1] - 1) + C.blue_[1] * (X + Z)) -
       C.blue_[0] * (Y * (C.red_[1] - 1) + C.red_[1] * (X + Z))) /
      d;

  float Sb =
      (X * (C.green_[1] - C.red_[1]) - C.red_[0] * (Y * (C.green_[1] - 1) + C.green_[1] * (X + Z)) +
       C.green_[0] * (Y * (C.red_[1] - 1) + C.red_[1] * (X + Z))) /
      d;
  cv::Matx33f M(Sr * C.red_[0], Sr * C.red_[1], Sr * (1.f - C.red_[0] - C.red_[1]), Sg * C.green_[0],
                Sg * C.green_[1], Sg * (1.f - C.green_[0] - C.green_[1]), Sb * C.blue_[0],
                Sb * C.blue_[1], Sb * (1.f - C.blue_[0] - C.blue_[1]));
  return M;
}

inline cv::Matx33f XYZ_TO_RGB_f33(const ColorSpacePrimaries& C, float Y = 1.0f) {
  cv::Matx33f M = RGB_TO_XYZ_f33(C, Y);
  return M.inv();
}

// Gamut compression constants
const float               smooth_cusps           = 0.12f;
const float               smooth_J               = 0.0f;  // could be eliminated
const float               smooth_M               = 0.27f;
const float               cusp_mid_blend         = 1.3f;

const float               focus_gain_blend       = 0.3f;
const float               focus_adjust_gain      = 0.55f;
const float               focus_distance         = 1.35f;
const float               focus_distance_scaling = 1.75f;

// CAM Parameters
const float               L_A                    = 100.f;
const float               Y_b                    = 20.f;

const float               ac_resp                = 1.f;
const float               ra                     = 2.f;
const float               ba                     = 0.05f + (2.f - ra);

const float               surround[3] = {0.9f, 0.59f, 0.9f};  // Average viewing condition

const ColorSpacePrimaries CAM16_PRI   = {
    {0.8336f, 0.1735f}, {2.3854f, -1.4659f}, {0.087f, -0.125f}, {0.333f, 0.333f}};

const cv::Matx33f CAM16_RGB_TO_XYZ = RGB_TO_XYZ_f33(CAM16_PRI, 1.f);

inline cv::Matx33f       GeneratePanlrcm(float _ra = 2.f, float _ba = 0.05f) {
  cv::Matx33f panlrcm_data = {_ra,        1.f, 1.f / 9.f,  1.f,       -12.f / 11.f,
                                    1.f / 9.f, _ba,  1.f / 11.f, -2.f / 9.f};
  cv::Matx33f panlrcm      = panlrcm_data.inv();

  for (int i = 0; i < 3; ++i) {
    float n       = 460.f / panlrcm(0, i);
    panlrcm(0, i) = panlrcm(0, i) * n;
    panlrcm(1, i) = panlrcm(1, i) * n;
    panlrcm(2, i) = panlrcm(2, i) * n;
  }

  return panlrcm;
}

struct TSParams {
  float n_;
  float n_r_;
  float g_;
  float t_1_;
  float c_t_;
  float s_2_;
  float u_2_;
  float m_2_;
};

const cv::Matx33f PANLRCM = GeneratePanlrcm(ra, ba);

struct ODTParams {
  float       peak_luminance_;

  // Tone scale, set via TSParams structure
  TSParams    ts_params_;

  float       focus_dist_;

  // Chroma Compression
  float       limit_J_max_;
  float       mid_J_;
  float       model_gamma_;
  float       sat_;
  float       sat_thr_;
  float       compr_;

  // Limit
  cv::Matx33f LIMIT_RGB_TO_XYZ_;
  cv::Matx33f LIMIT_XYZ_TO_RGB_;
  cv::Matx13f XYZ_w_limit_;

  // Output
  cv::Matx33f OUTPUT_RGB_TO_XYZ_;
  cv::Matx33f OUTPUT_XYZ_TO_RGB_;
  cv::Matx13f XYZ_w_output_;

  float       lower_hull_gamma_;
};

inline float signum(float x) {
  if (x > 0.f) {
    return 1.f;
  } else if (x < 0.f) {
    return -1.f;
  } else {
    return 0.f;
  }
}

inline cv::Matx13f mult_f3_f33(const cv::Matx13f& v, const cv::Matx33f& m) {
  return cv::Matx13f(v(0) * m(0, 0) + v(1) * m(1, 0) + v(2) * m(2, 0),
                     v(0) * m(0, 1) + v(1) * m(1, 1) + v(2) * m(2, 1),
                     v(0) * m(0, 2) + v(1) * m(1, 2) + v(2) * m(2, 2));
}

inline float Y_to_Hellwig_J(float Y, float _surround = 0.59f, float _L_A = 100.f, float _Y_b = 20.f) {
  float k   = 1.f / (5.f * _L_A + 1.f);
  float k4  = k * k * k * k;
  float F_L = 0.2f * k4 * (5.f * _L_A) + 0.1f * powf((1.f - k4), 2.f) * powf((5.f * _L_A), 1.f / 3.f);
  float n   = _Y_b / 100.f;
  float z   = 1.48f + sqrtf(n);
  float F_L_W = powf(F_L, 0.42f);
  float A_w   = (400.f * F_L_W) / (F_L_W + 27.13f);

  float F_L_Y = powf(F_L * fabsf(Y) / 100.f, 0.42f);

  return signum(Y) * 100.f * powf(((400.f * F_L_Y) / (27.13f + F_L_Y)) / A_w, _surround * z);
}
}  // namespace ColorUtils
};  // namespace puerhlab