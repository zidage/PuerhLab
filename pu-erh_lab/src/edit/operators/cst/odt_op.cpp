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

// [WIP] New operator ODT implementation closer to ACES workflow

#include "edit/operators/cst/odt_op.hpp"

#include <memory>
#include <opencv2/core.hpp>

#include "edit/operators/utils/color_utils.hpp"

namespace puerhlab {
ColorUtils::ColorSpace ACES_ODT_Op::ParseColorSpace(const std::string& cs_str) {
  if (cs_str == "rec709") {
    return ColorUtils::ColorSpace::REC709;
  } else if (cs_str == "rec2020") {
    return ColorUtils::ColorSpace::REC2020;
  } else if (cs_str == "p3_d65") {
    return ColorUtils::ColorSpace::P3_D65;
  } else if (cs_str == "prophoto") {
    return ColorUtils::ColorSpace::PROPHOTO;
  } else if (cs_str == "adobe_rgb") {
    return ColorUtils::ColorSpace::ADOBE_RGB;
  } else {
    return ColorUtils::ColorSpace::REC709;  // Default to Rec.709
  }
};

ColorUtils::ETOF ACES_ODT_Op::ParseETOF(const std::string& etof_str) {
  if (etof_str == "linear") {
    return ColorUtils::ETOF::LINEAR;
  } else if (etof_str == "st2084") {
    return ColorUtils::ETOF::ST2084;
  } else if (etof_str == "hlg") {
    return ColorUtils::ETOF::HLG;
  } else if (etof_str == "gamma_2_6") {
    return ColorUtils::ETOF::GAMMA_2_6;
  } else if (etof_str == "bt1886") {
    return ColorUtils::ETOF::BT1886;
  } else if (etof_str == "gamma_2_2") {
    return ColorUtils::ETOF::GAMMA_2_2;
  } else if (etof_str == "gamma_1_8") {
    return ColorUtils::ETOF::GAMMA_1_8;
  } else {
    return ColorUtils::ETOF::GAMMA_2_2;  // Default to Gamma 2.2
  }
};

std::string ACES_ODT_Op::ColorSpaceToString(ColorUtils::ColorSpace cs) {
  switch (cs) {
    case ColorUtils::ColorSpace::REC709:
      return "rec709";
    case ColorUtils::ColorSpace::REC2020:
      return "rec2020";
    case ColorUtils::ColorSpace::P3_D65:
      return "p3_d65";
    case ColorUtils::ColorSpace::PROPHOTO:
      return "prophoto";
    case ColorUtils::ColorSpace::ADOBE_RGB:
      return "adobe_rgb";
    default:
      return "rec709";
  }
}

std::string ACES_ODT_Op::ETOFToString(ColorUtils::ETOF etof) {
  switch (etof) {
    case ColorUtils::ETOF::LINEAR:
      return "linear";
    case ColorUtils::ETOF::ST2084:
      return "st2084";
    case ColorUtils::ETOF::HLG:
      return "hlg";
    case ColorUtils::ETOF::GAMMA_2_6:
      return "gamma_2_6";
    case ColorUtils::ETOF::BT1886:
      return "bt1886";
    case ColorUtils::ETOF::GAMMA_2_2:
      return "gamma_2_2";
    case ColorUtils::ETOF::GAMMA_1_8:
      return "gamma_1_8";
    default:
      return "gamma_2_2";
  }
}

ACES_ODT_Op::ACES_ODT_Op(const nlohmann::json& params) { SetParams(params); }

void ACES_ODT_Op::Apply(std::shared_ptr<ImageBuffer>) {
  // [WIP] Implement ODT application using odt_params_

  throw std::runtime_error(
      "ACES_ODT_Op: Use CST_Op with OCIO instead. This operator is used only as a descriptor for "
      "the ODT stage.");
}

auto ACES_ODT_Op::GetParams() const -> nlohmann::json {
  nlohmann::json o;
  nlohmann::json j;
  j["encoding_space"] = ColorSpaceToString(encoding_space_);
  j["encoding_etof"]  = ETOFToString(encoding_etof_);
  j["limiting_space"] = ColorSpaceToString(limiting_space_);
  j["peak_luminance"] = peak_luminance_;
  o["aces_odt"]       = j;
  return o;
}

void ACES_ODT_Op::SetParams(const nlohmann::json& in_j) {
  if (!in_j.contains("aces_odt")) {
    return;
  }
  const nlohmann::json& j = in_j["aces_odt"];

  if (j.contains("encoding_space")) {
    encoding_space_ = ParseColorSpace(j["encoding_space"].get<std::string>());
  }
  if (j.contains("encoding_etof")) {
    encoding_etof_ = ParseETOF(j["encoding_etof"].get<std::string>());
    // std::cout << static_cast<int>(encoding_etof_) << std::endl;
  }
  if (j.contains("limiting_space")) {
    limiting_space_ = ParseColorSpace(j["limiting_space"].get<std::string>());
  }
  if (j.contains("peak_luminance")) {
    peak_luminance_ = j["peak_luminance"].get<float>();
  }
  init_ODTParams();

  // to_output_params_.limit_matx_ =
  to_output_params_.limit_to_display_matx_ =
      ColorUtils::RGB_TO_XYZ_f33(ColorUtils::SpaceEnumToPrimary(limiting_space_)) *
      ColorUtils::XYZ_TO_RGB_f33(ColorUtils::SpaceEnumToPrimary(encoding_space_));

  to_output_params_.etof_ = encoding_etof_;
}

void ACES_ODT_Op::SetGlobalParams(OperatorParams& global_params) const {
  // ODT operator does not modify global params
  global_params.to_output_params_ = to_output_params_;
  global_params.to_output_dirty_  = true;
}

ColorUtils::JMhParams init_JMhParams(const ColorUtils::ColorSpacePrimaries& prims) {
  using namespace ColorUtils;

  const cv::Matx33f RGB_to_XYZ = RGB_TO_XYZ_f33(prims, 1.f);
  const cv::Matx13f XYZ_w      = cv::Matx13f(ref_lum, ref_lum, ref_lum) * RGB_to_XYZ;

  float             Y_w        = XYZ_w(1);

  // Step 0: Converting CIE XYZ tristimulus values to sharpened RGB values
  cv::Matx13f       RGW_w      = XYZ_w * MATRIX_16;

  // Viewing condition dependent parameters
  const float       k          = 1.f / (5.f * L_A + 1.f);
  const float       k4         = k * k * k * k;
  const float       F_L =
      0.2f * k4 * (5.f * L_A) + 0.1f * powf(1.f - k4, 2.f) * powf(5.f * L_A, 1.f / 3.f);

  const float       F_L_n  = F_L / ref_lum;
  const float       cz     = model_gamma;

  const cv::Matx13f D_RGB  = {F_L_n * Y_w / RGW_w(0), F_L_n * Y_w / RGW_w(1),
                              F_L_n * Y_w / RGW_w(2)};

  const cv::Matx13f RGB_wc = {D_RGB(0) * RGW_w(0), D_RGB(1) * RGW_w(1), D_RGB(2) * RGW_w(2)};

  const cv::Matx13f RGB_Aw = {pacrc_fwd(RGB_wc(0)), pacrc_fwd(RGB_wc(1)), pacrc_fwd(RGB_wc(2))};

  cv::Matx33f       cone_response_to_Aab =
      cv::Matx33f::diag({cam_nl_scale, cam_nl_scale, cam_nl_scale}) * base_cone_repponse_to_Aab;
  float A_w = cone_response_to_Aab(0, 0) * RGB_Aw(0) + cone_response_to_Aab(1, 0) * RGB_Aw(1) +
              cone_response_to_Aab(2, 0) * RGB_Aw(2);

  float       A_w_J               = _pacrc_fwd_(F_L);

  // Prescale the CAM16 LMS responses to directly provide for chromatic adaptation
  cv::Matx33f M1                  = RGB_to_XYZ * MATRIX_16;
  cv::Matx33f M2                  = cv::Matx33f::diag({ref_lum, ref_lum, ref_lum});
  cv::Matx33f MATRIX_RGB_to_CAM16 = M1 * M2;
  cv::Matx33f MATRIX_RGB_to_CAM16_c =
      MATRIX_RGB_to_CAM16 * cv::Matx33f::diag({D_RGB(0), D_RGB(1), D_RGB(2)});

  cv::Matx33f MATRIX_cone_response_to_Aab = {cone_response_to_Aab(0, 0) / A_w,
                                             cone_response_to_Aab(0, 1) * 43.f * surround[2],
                                             cone_response_to_Aab(0, 2) * 43.f * surround[2],
                                             cone_response_to_Aab(1, 0) / A_w,
                                             cone_response_to_Aab(1, 1) * 43.f * surround[2],
                                             cone_response_to_Aab(1, 2) * 43.f * surround[2],
                                             cone_response_to_Aab(2, 0) / A_w,
                                             cone_response_to_Aab(2, 1) * 43.f * surround[2],
                                             cone_response_to_Aab(2, 2) * 43.f * surround[2]};

  JMhParams   p;
  p.MATRIX_RGB_to_CAM16_c_       = MATRIX_RGB_to_CAM16_c;
  p.MATRIX_CAM16_c_to_RGB_       = MATRIX_RGB_to_CAM16_c.inv();
  p.MATRIX_cone_response_to_Aab_ = MATRIX_cone_response_to_Aab;
  p.MATRIX_Aab_to_cone_response_ = MATRIX_cone_response_to_Aab.inv();
  p.F_L_n_                       = F_L_n;
  p.cz_                          = cz;
  p.inv_cz_                      = 1.f / cz;
  // NOTE: CTL uses JMhParams.A_w_J = _pacrc_fwd(F_L). This value is required
  // by the achromatic optimization path (J_to_Y / Y_to_J).
  // Historically this field was named A_w_z_ in our CPU struct, but for the
  // CTL port (and GPU path) it must store A_w_J.
  p.A_w_z_                       = A_w_J;
  p.inv_A_w_J_                   = 1.f / A_w_J;

  return p;
}

void ACES_ODT_Op::init_JMhParams() {
  to_output_params_.odt_params_.input_params_ = ::puerhlab::init_JMhParams(ColorUtils::AP0_PRIMARY);
  to_output_params_.odt_params_.reach_params_ =
      ::puerhlab::init_JMhParams(ColorUtils::REACH_PRIMARY);
  to_output_params_.odt_params_.limit_params_ =
      ::puerhlab::init_JMhParams(ColorUtils::SpaceEnumToPrimary(limiting_space_));
  auto prims = ColorUtils::SpaceEnumToPrimary(limiting_space_);
  std::cout << "Limiting space primaries:\n"
            << "[" << prims.red_[0] << ", " << prims.red_[1] << "]\n"
            << "[" << prims.green_[0] << ", " << prims.green_[1] << "]\n"
            << "[" << prims.blue_[0] << ", " << prims.blue_[1] << "]\n"
            << "[" << prims.white_[0] << ", " << prims.white_[1] << "]"
            << std::endl;
}

void ACES_ODT_Op::init_TSParams() {
  const float n         = peak_luminance_;

  const float n_r       = 100.0f;   // normalized white in nits (what 1.0 should be)
  const float g         = 1.15f;    // surround / contrast
  const float c         = 0.18f;    // anchor for 18% grey
  const float c_d       = 10.013f;  // output luminance of 18% grey (in nits)
  const float w_g       = 0.14f;    // change in grey between different peak luminance
  const float t_1       = 0.04f;    // shadow toe or flare/glare compensation
  const float r_hit_min = 128.f;    // scene-referred value "hitting the roof"
  const float r_hit_max = 896.f;    // scene-referred value "hitting the roof"

  // Calculate output constants
  const float r_hit  = r_hit_min + (r_hit_max - r_hit_min) * (log(n / n_r) / log(10000.f / 100.f));
  const float m_0    = (n / n_r);
  const float m_1    = 0.5f * (m_0 + sqrt(m_0 * (m_0 + 4.f * t_1)));
  const float u      = pow((r_hit / m_1) / ((r_hit / m_1) + 1), g);
  const float m      = m_1 / u;
  const float w_i    = log(n / 100.f) / log(2.f);
  const float c_t    = c_d / n_r * (1.f + w_i * w_g);
  const float g_ip   = 0.5f * (c_t + sqrt(c_t * (c_t + 4.f * t_1)));
  const float g_ipp2 = -(m_1 * pow((g_ip / m), (1.f / g))) / (pow(g_ip / m, 1.f / g) - 1.f);
  const float w_2    = c / g_ipp2;
  const float s_2    = w_2 * m_1;
  const float u_2    = pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
  const float m_2    = m_1 / u_2;

  ColorUtils::TSParams TonescaleParams = {
      n, n_r, g, t_1, c_t, s_2, u_2, m_2, 8.f * r_hit, n / (u_2 * n_r), log10(n / n_r)};
  // std::cout << "r_hit: " << TonescaleParams.forward_limit_ << std::endl;
  to_output_params_.odt_params_.ts_params_ = TonescaleParams;
}

void ACES_ODT_Op::init_ODTParams() {
  using namespace ColorUtils;
  to_output_params_.odt_params_.peak_luminance_ = peak_luminance_;
  init_JMhParams();
  init_TSParams();
  TSParams&  ts        = to_output_params_.odt_params_.ts_params_;
  ODTParams& odt       = to_output_params_.odt_params_;
  // Shared compression parameters
  odt.limit_J_max_     = Y_to_J(peak_luminance_, odt.input_params_);
  odt.model_gamma_inv_ = 1.f / model_gamma;
  odt.table_reach_M_ =
      MakeReachMTable(to_output_params_.odt_params_.reach_params_, odt.limit_J_max_);

  // Chroma compression parameters
  odt.sat_     = fmaxf(0.2f, chroma_expand - (chroma_expand * chroma_expand_fact) * ts.log_peak_);
  odt.sat_thr_ = chroma_expand_thr / peak_luminance_;
  odt.compr_   = chroma_compress + (chroma_compress * chroma_compress_fact) * ts.log_peak_;
  odt.chroma_compress_scale_ = powf(0.03379f * peak_luminance_, 0.30596f) - 0.45135f;

  // Gamut compression parameters
  odt.mid_J_                 = Y_to_J(ts.c_t_ * ref_lum, odt.input_params_);
  odt.focus_dist_ = focus_distance + focus_distance * focus_distance_scaling * ts.log_peak_;
  const float lower_hull_gama = 1.14f + 0.07f * ts.log_peak_;
  odt.lower_hull_gamma_       = lower_hull_gama;
  odt.lower_hull_gamma_inv_   = 1.f / lower_hull_gama;
  odt.table_gamut_cusps_      = MakeUniformHueGamutTable(odt.reach_params_, odt.limit_params_, odt);

  odt.table_hues_             = std::make_shared<std::array<float, TOTAL_TABLE_SIZE>>();
  for (int i = 0; i < TOTAL_TABLE_SIZE; ++i) {
    (*odt.table_hues_)[i] = (*odt.table_gamut_cusps_)[i](2);
  }
  odt.table_upper_hull_gammas_ = std::make_shared<std::array<float, TOTAL_TABLE_SIZE>>(
      MakeUpperHullGammaTable(*odt.table_gamut_cusps_, odt));
  odt.hue_linearity_search_range_ = DetermineHueLinearitySearchRange(*odt.table_hues_);
}
};  // namespace puerhlab