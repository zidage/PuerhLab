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
  return j;
}

void ACES_ODT_Op::SetParams(const nlohmann::json& j) {
  if (j.contains("encoding_space")) {
    encoding_space_ = ParseColorSpace(j["encoding_space"].get<std::string>());
  }
  if (j.contains("encoding_etof")) {
    encoding_etof_ = ParseETOF(j["encoding_etof"].get<std::string>());
  }
  if (j.contains("limiting_space")) {
    limiting_space_ = ParseColorSpace(j["limiting_space"].get<std::string>());
  }
  if (j.contains("peak_luminance")) {
    peak_luminance_ = j["peak_luminance"].get<float>();
  }
  init_ODTParams();
}

void ACES_ODT_Op::SetGlobalParams(OperatorParams& global_params) const {
  // ODT operator does not modify global params
  global_params.odt_params_ = odt_params_;
}

void ACES_ODT_Op::init_TSParams() {
  const float          n         = peak_luminance_;

  const float          n_r       = 100.0;   // normalized white in nits (what 1.0 should be)
  const float          g         = 1.15;    // surround / contrast
  const float          c         = 0.18;    // anchor for 18% grey
  const float          c_d       = 10.013;  // output luminance of 18% grey (in nits)
  const float          w_g       = 0.14;    // change in grey between different peak luminance
  const float          t_1       = 0.04;    // shadow toe or flare/glare compensation
  const float          r_hit_min = 128.;    // scene-referred value "hitting the roof"
  const float          r_hit_max = 896.;    // scene-referred value "hitting the roof"

  // Calculate output constants
  const float          r_hit     = r_hit_min + r_hit_max * (log(n / n_r) / log(10000. / 100.));
  const float          m_0       = (n / n_r);
  const float          m_1       = 0.5 * (m_0 + sqrt(m_0 * (m_0 + 4. * t_1)));
  const float          u         = pow((r_hit / m_1) / ((r_hit / m_1) + 1), g);
  const float          m         = m_1 / u;
  const float          w_i       = log(n / 100.) / log(2.);
  const float          c_t       = c_d / n_r * (1. + w_i * w_g);
  const float          g_ip      = 0.5 * (c_t + sqrt(c_t * (c_t + 4. * t_1)));
  const float          g_ipp2 = -(m_1 * pow((g_ip / m), (1. / g))) / (pow(g_ip / m, 1. / g) - 1.);
  const float          w_2    = c / g_ipp2;
  const float          s_2    = w_2 * m_1;
  const float          u_2    = pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
  const float          m_2    = m_1 / u_2;

  ColorUtils::TSParams TonescaleParams = {n, n_r, g, t_1, c_t, s_2, u_2, m_2};
  odt_params_.ts_params_               = std::move(TonescaleParams);
}

void ACES_ODT_Op::init_ODTParams() {
  using namespace ColorUtils;
  odt_params_.peak_luminance_ = peak_luminance_;
  init_TSParams();
  TSParams&   ts                   = odt_params_.ts_params_;
  ODTParams&  odt                  = odt_params_;

  float       limit_J_max          = Y_to_Hellwig_J(odt_params_.peak_luminance_);
  float       mid_J                = Y_to_Hellwig_J(odt_params_.ts_params_.c_t_ * 100.f);

  // Chroma compress presets
  const float chroma_compress      = 2.4;
  const float chroma_compress_fact = 3.3;
  const float chroma_expand        = 1.3;
  const float chroma_expand_fact   = 0.69;
  const float chroma_expand_thr    = 0.5;

  const float log_peak             = (ts.n_ / ts.n_r_);
  const float compr   = chroma_compress + (chroma_compress * chroma_compress_fact) * log_peak;
  const float sat     = fmaxf(0.2, chroma_expand - (chroma_expand * chroma_expand_fact) * log_peak);
  const float sat_thr = chroma_expand_thr / ts.n_;

  const float model_gamma = 1.f / (surround[1] * (1.48f + sqrtf(Y_b / L_A)));

  const float focus_dist  = focus_distance + focus_distance * focus_distance_scaling * log_peak;

  const cv::Matx13f RGB_w = {100.f, 100.f, 100.f};

  // Input Primaries (AP0)
  const cv::Matx33f INPUT_RGB_TO_XYZ  = RGB_TO_XYZ_f33(AP0_PRIMARY, 1.f);
  const cv::Matx33f INPUT_XYZ_TO_RGB  = INPUT_RGB_TO_XYZ.inv();
  cv::Matx13f       XYZ_w_in          = RGB_w * INPUT_RGB_TO_XYZ;

  const float       lower_hull_gamma  = 1.14f;

  // Limiting Primaries
  const cv::Matx33f LIMIT_RGB_TO_XYZ  = RGB_TO_XYZ_f33(SpaceEnumToPrimary(limiting_space_), 1.f);
  const cv::Matx33f LIMIT_XYZ_TO_RGB  = LIMIT_RGB_TO_XYZ.inv();
  cv::Matx13f       XYZ_w_limit       = RGB_w * LIMIT_RGB_TO_XYZ;

  const cv::Matx33f OUTPUT_RGB_TO_XYZ = RGB_TO_XYZ_f33(SpaceEnumToPrimary(encoding_space_), 1.f);
  const cv::Matx33f OUTPUT_XYZ_TO_RGB = OUTPUT_RGB_TO_XYZ.inv();
  cv::Matx13f       XYZ_w_output      = RGB_w * OUTPUT_RGB_TO_XYZ;

  odt.peak_luminance_                 = peak_luminance_;
  odt.limit_J_max_                    = limit_J_max;
  odt.mid_J_                          = mid_J;
  odt.model_gamma_                    = model_gamma;
  odt.sat_                            = sat;
  odt.sat_thr_                        = sat_thr;
  odt.compr_                          = compr;

  odt.focus_dist_                     = focus_dist;

  odt.LIMIT_RGB_TO_XYZ_               = LIMIT_RGB_TO_XYZ;
  odt.LIMIT_XYZ_TO_RGB_               = LIMIT_XYZ_TO_RGB;
  odt.XYZ_w_limit_                    = XYZ_w_limit;

  // Output Primaries
  odt.OUTPUT_RGB_TO_XYZ_              = OUTPUT_RGB_TO_XYZ;
  odt.OUTPUT_XYZ_TO_RGB_              = OUTPUT_XYZ_TO_RGB;
  odt.XYZ_w_output_                   = XYZ_w_output;

  odt.lower_hull_gamma_               = lower_hull_gamma;
}
};  // namespace puerhlab