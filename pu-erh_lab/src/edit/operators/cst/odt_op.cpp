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

#include "edit/operators/cst/odt_op.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <string>
#include <tuple>
#include <opencv2/core.hpp>
#include <unordered_map>

#include "edit/operators/utils/color_utils.hpp"

namespace puerhlab {
namespace {
using ColorSpace            = ColorUtils::ColorSpace;
using ETOF                  = ColorUtils::ETOF;
using OutputTransformMethod = ColorUtils::OutputTransformMethod;

struct ODTCacheKey {
  ColorSpace    encoding_space_      = ColorSpace::REC709;
  ColorSpace    limiting_space_      = ColorSpace::REC709;
  std::uint32_t peak_luminance_bits_ = 0;

  auto operator==(const ODTCacheKey& other) const -> bool {
    return encoding_space_ == other.encoding_space_ &&
           limiting_space_ == other.limiting_space_ &&
           peak_luminance_bits_ == other.peak_luminance_bits_;
  }
};

struct ODTCacheKeyHash {
  auto operator()(const ODTCacheKey& key) const noexcept -> std::size_t {
    std::size_t h = std::hash<int>{}(static_cast<int>(key.encoding_space_));
    h ^= std::hash<int>{}(static_cast<int>(key.limiting_space_)) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint32_t>{}(key.peak_luminance_bits_) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

auto ODTParamsCache()
    -> std::unordered_map<ODTCacheKey, ColorUtils::ACESODTParams, ODTCacheKeyHash>& {
  static std::unordered_map<ODTCacheKey, ColorUtils::ACESODTParams, ODTCacheKeyHash> cache;
  return cache;
}

auto ODTParamsCacheMutex() -> std::mutex& {
  static std::mutex cache_mutex;
  return cache_mutex;
}

auto BuildODTCacheKey(ColorSpace encoding_space, ColorSpace limiting_space, float peak_luminance)
    -> ODTCacheKey {
  return ODTCacheKey{
      encoding_space,
      limiting_space,
      std::bit_cast<std::uint32_t>(peak_luminance),
  };
}

auto CanonicalizeToken(std::string value) -> std::string {
  for (char& ch : value) {
    if (ch == '-' || ch == ' ' || ch == '/' || ch == '.') {
      ch = '_';
    } else {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  return value;
}

void MergeJson(nlohmann::json& dst, const nlohmann::json& src) {
  if (!src.is_object()) {
    dst = src;
    return;
  }

  for (auto it = src.begin(); it != src.end(); ++it) {
    if (dst.contains(it.key()) && dst[it.key()].is_object() && it.value().is_object()) {
      MergeJson(dst[it.key()], it.value());
    } else {
      dst[it.key()] = it.value();
    }
  }
}

auto DefaultOpenDRTAuthoringParams() -> nlohmann::json {
  return {
      {"look_preset", "standard"},
      {"tonescale_preset", "use_look_preset"},
      {"display_encoding_preset", "srgb_display"},
      {"creative_white_preset", "use_look_preset"},
      {"tn_Lp", 100.0f},
      {"tn_gb", 0.13f},
      {"pt_hdr", 0.5f},
      {"tn_Lg", 10.0f},
  };
}

auto DefaultAuthoringParams() -> nlohmann::json {
  return {
      {"method", "open_drt"},
      {"encoding_space", "rec709"},
      {"encoding_etof", "gamma_2_2"},
      {"limiting_space", "rec709"},
      {"peak_luminance", 100.0f},
      {"open_drt", DefaultOpenDRTAuthoringParams()},
  };
}

auto NormalizeLookPreset(const std::string& preset) -> std::string {
  const std::string token = CanonicalizeToken(preset);
  return token.empty() ? "standard" : token;
}

auto NormalizeTonescalePreset(const std::string& preset) -> std::string {
  std::string token = CanonicalizeToken(preset);
  if (token == "use_look" || token == "look_preset") {
    token = "use_look_preset";
  }
  return token.empty() ? "use_look_preset" : token;
}

auto NormalizeDisplayEncodingPreset(const std::string& preset) -> std::string {
  std::string token = CanonicalizeToken(preset);
  if (token == "srgb") token = "srgb_display";
  if (token == "displayp3") token = "display_p3";
  if (token == "p3d60") token = "p3_d60";
  if (token == "p3dci") token = "p3_dci";
  if (token == "rec2100pq") token = "rec2100_pq";
  if (token == "rec2100hlg") token = "rec2100_hlg";
  if (token == "dolbypq") token = "dolby_pq";
  return token.empty() ? "srgb_display" : token;
}

auto NormalizeCreativeWhitePreset(const std::string& preset) -> std::string {
  std::string token = CanonicalizeToken(preset);
  if (token == "use_look") token = "use_look_preset";
  return token.empty() ? "use_look_preset" : token;
}

void OverrideFloat(const nlohmann::json& j, const char* key, float& value) {
  if (j.contains(key) && j[key].is_number()) {
    value = j[key].get<float>();
  }
}

void OverrideBoolLike(const nlohmann::json& j, const char* key, int& value) {
  if (!j.contains(key)) {
    return;
  }

  if (j[key].is_boolean()) {
    value = j[key].get<bool>() ? 1 : 0;
  } else if (j[key].is_number_integer()) {
    value = j[key].get<int>() != 0 ? 1 : 0;
  }
}

auto CompressToeQuadratic(float x, float toe, bool inverse) -> float {
  if (toe == 0.0f) return x;
  if (!inverse) {
    return ColorUtils::signum(x) * ((x * x) / (std::abs(x) + toe));
  }

  const float abs_x = std::abs(x);
  const float value = (abs_x + std::sqrt(abs_x * (4.0f * toe + abs_x))) * 0.5f;
  return std::copysign(value, x);
}

auto MakeWhiteXYZFromXY(const cv::Vec2f& xy) -> cv::Vec3f {
  const float X = xy[0] / xy[1];
  const float Y = 1.0f;
  const float Z = (1.0f - xy[0] - xy[1]) / xy[1];
  return {X, Y, Z};
}

auto BradfordAdaptationRow(const cv::Vec2f& src_xy, const cv::Vec2f& dst_xy) -> cv::Matx33f {
  if (cv::norm(src_xy - dst_xy) <= 1e-6f) {
    return cv::Matx33f::eye();
  }

  const cv::Matx33f bradford_col = {
      0.8951f, 0.2664f, -0.1614f,
      -0.7502f, 1.7135f, 0.0367f,
      0.0389f, -0.0685f, 1.0296f,
  };
  const cv::Matx33f bradford_col_inv = bradford_col.inv();

  const cv::Vec3f src_xyz = MakeWhiteXYZFromXY(src_xy);
  const cv::Vec3f dst_xyz = MakeWhiteXYZFromXY(dst_xy);

  const cv::Vec3f src_lms = bradford_col * src_xyz;
  const cv::Vec3f dst_lms = bradford_col * dst_xyz;

  const cv::Matx33f scale = {
      dst_lms[0] / src_lms[0], 0.0f, 0.0f,
      0.0f, dst_lms[1] / src_lms[1], 0.0f,
      0.0f, 0.0f, dst_lms[2] / src_lms[2],
  };
  const cv::Matx33f adapt_col = bradford_col_inv * scale * bradford_col;
  return adapt_col.t();
}

auto WhitePointXYFromMode(int mode) -> cv::Vec2f {
  switch (mode) {
    case 0:
      return {0.28300f, 0.29700f};
    case 1:
      return {0.29903f, 0.31488f};
    case 2:
      return {0.31270f, 0.32900f};
    case 3:
      return {0.32168f, 0.33767f};
    case 4:
      return {0.33243f, 0.34744f};
    case 5:
      return {0.34570f, 0.35850f};
    default:
      return {0.31270f, 0.32900f};
  }
}

auto LimitingSpaceWhiteXY(ColorSpace limiting_space) -> cv::Vec2f {
  switch (limiting_space) {
    case ColorSpace::P3_D60:
      return {0.32168f, 0.33767f};
    case ColorSpace::P3_DCI:
    case ColorSpace::XYZ:
      return {0.31400f, 0.35100f};
    default:
      return {0.31270f, 0.32900f};
  }
}

auto MakeRenderToLimitMatrix(ColorSpace limiting_space, const cv::Vec2f& target_white_xy)
    -> cv::Matx33f {
  const cv::Matx33f render_to_xyz = ColorUtils::RGB_TO_XYZ_f33(ColorSpace::P3_D65);
  const cv::Matx33f adapt =
      BradfordAdaptationRow(cv::Vec2f(0.31270f, 0.32900f), target_white_xy);
  const cv::Matx33f xyz_to_limit = ColorUtils::XYZ_TO_RGB_f33(limiting_space);
  return render_to_xyz * adapt * xyz_to_limit;
}

auto WhiteNormFromRenderToLimit(const cv::Matx33f& render_to_limit) -> float {
  const cv::Matx13f unit_render(1.0f, 1.0f, 1.0f);
  const cv::Matx13f limit_white = ColorUtils::mult_f3_f33(unit_render, render_to_limit);
  const float       max_channel = std::max(limit_white(0), std::max(limit_white(1), limit_white(2)));
  return max_channel > 1e-6f ? 1.0f / max_channel : 1.0f;
}

auto DisplayPresetToCommonFields(const std::string& preset)
    -> std::tuple<ColorSpace, ETOF, ColorSpace> {
  const std::string token = NormalizeDisplayEncodingPreset(preset);
  if (token == "rec1886") return {ColorSpace::REC709, ETOF::BT1886, ColorSpace::REC709};
  if (token == "display_p3") return {ColorSpace::P3_D65, ETOF::GAMMA_2_2, ColorSpace::P3_D65};
  if (token == "p3_d60") return {ColorSpace::P3_D60, ETOF::GAMMA_2_6, ColorSpace::P3_D60};
  if (token == "p3_dci") return {ColorSpace::P3_DCI, ETOF::GAMMA_2_6, ColorSpace::P3_DCI};
  if (token == "xyz") return {ColorSpace::XYZ, ETOF::GAMMA_2_6, ColorSpace::XYZ};
  if (token == "rec2100_pq") return {ColorSpace::REC2020, ETOF::ST2084, ColorSpace::REC2020};
  if (token == "rec2100_hlg") return {ColorSpace::REC2020, ETOF::HLG, ColorSpace::REC2020};
  if (token == "dolby_pq") return {ColorSpace::P3_D65, ETOF::ST2084, ColorSpace::P3_D65};
  return {ColorSpace::REC709, ETOF::GAMMA_2_2, ColorSpace::REC709};
}

auto DetermineOpenDRTSurroundComp(ColorSpace encoding_space, ETOF etof) -> int {
  if (etof == ETOF::BT1886) {
    return 1;
  }
  if (etof == ETOF::GAMMA_2_2 &&
      (encoding_space == ColorSpace::REC709 || encoding_space == ColorSpace::P3_D65)) {
    return 2;
  }
  return 0;
}

auto DetermineDisplayLinearScale(ETOF etof) -> float {
  return (etof == ETOF::ST2084 || etof == ETOF::HLG) ? 100.0f : 1.0f;
}

auto ResolveACESLimitingSpace(ColorSpace limiting_space) -> ColorSpace {
  return limiting_space == ColorSpace::XYZ ? ColorSpace::P3_DCI : limiting_space;
}

void ApplyLookPreset(const std::string& preset, ColorUtils::OpenDRTResolvedParams& p) {
  p = ColorUtils::OpenDRTResolvedParams{};
  const std::string token = NormalizeLookPreset(preset);
  if (token == "arriba") {
    p.tn_con_ = 1.05f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.1f, p.tn_off_ = 0.01f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.5f, p.tn_lcon_w_ = 0.2f;
    p.creative_white_mode_ = 2, p.creative_white_luminance_mix_ = 0.25f;
    p.pt_lml_r_ = 0.45f, p.pt_lmh_r_ = 0.25f;
    p.ptm_low_ = 1.0f, p.ptm_low_rng_ = 0.4f, p.ptm_high_rng_ = 0.66f, p.ptm_high_st_ = 0.6f;
    p.brlp_ = 0.0f, p.brlp_r_ = -1.7f, p.brlp_g_ = -2.0f, p.brlp_b_ = -0.5f;
    p.hs_r_rng_ = 0.8f, p.hs_c_ = 0.15f;
    return;
  }
  if (token == "sylvan") {
    p.tn_con_ = 1.6f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.01f, p.tn_off_ = 0.01f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.25f, p.tn_lcon_w_ = 0.75f;
    p.rs_sa_ = 0.25f;
    p.pt_lml_ = 0.15f, p.pt_lml_g_ = 0.15f, p.pt_lmh_r_ = 0.15f, p.pt_lmh_b_ = 0.15f;
    p.ptl_c_ = 0.05f, p.ptl_y_ = 0.05f;
    p.ptm_low_ = 0.5f, p.ptm_low_rng_ = 0.5f, p.ptm_high_rng_ = 0.5f, p.ptm_high_st_ = 0.5f;
    p.brl_ = -1.0f, p.brl_r_ = -2.0f, p.brl_g_ = -2.0f, p.brl_b_ = 0.0f, p.brl_rng_ = 0.25f;
    p.brl_st_ = 0.25f;
    p.brlp_ = -1.0f, p.brlp_r_ = -0.5f, p.brlp_g_ = -0.25f, p.brlp_b_ = -0.25f;
    p.hc_r_rng_ = 0.4f;
    p.hs_r_rng_ = 1.15f, p.hs_g_ = 0.8f, p.hs_g_rng_ = 1.25f, p.hs_b_ = 0.6f;
    p.hs_c_ = 0.25f, p.hs_c_rng_ = 0.25f, p.hs_m_ = 0.25f, p.hs_m_rng_ = 0.5f;
    p.hs_y_ = 0.35f, p.hs_y_rng_ = 0.5f;
    return;
  }
  if (token == "colorful") {
    p.tn_con_ = 1.5f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.003f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.4f, p.tn_lcon_w_ = 0.5f;
    p.pt_lml_ = 0.5f, p.pt_lml_r_ = 1.0f, p.pt_lml_b_ = 0.5f;
    p.pt_lmh_ = 0.15f, p.pt_lmh_r_ = 0.15f, p.pt_lmh_b_ = 0.15f;
    p.ptl_c_ = 0.05f, p.ptl_m_ = 0.06f, p.ptl_y_ = 0.05f;
    p.ptm_low_ = 0.8f, p.ptm_low_rng_ = 0.5f, p.ptm_low_st_ = 0.4f, p.ptm_high_rng_ = 0.4f;
    p.brl_r_ = -1.25f, p.brl_g_ = -1.25f, p.brl_b_ = -0.25f, p.brl_rng_ = 0.3f, p.brl_st_ = 0.5f;
    p.brlp_b_ = -0.5f;
    p.hc_r_rng_ = 0.4f;
    p.hs_r_ = 0.5f, p.hs_r_rng_ = 0.8f, p.hs_b_ = 0.5f;
    p.hs_y_ = 0.25f;
    return;
  }
  if (token == "aery") {
    p.tn_con_ = 1.15f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.04f, p.tn_off_ = 0.006f;
    p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 0.5f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.5f, p.tn_lcon_w_ = 2.0f;
    p.creative_white_mode_ = 1;
    p.rs_sa_ = 0.25f, p.rs_rw_ = 0.2f, p.rs_bw_ = 0.5f;
    p.pt_lml_ = 0.0f, p.pt_lml_g_ = 0.15f, p.pt_lmh_ = 0.0f, p.pt_lmh_r_ = 0.1f;
    p.ptl_c_ = 0.05f, p.ptl_y_ = 0.05f;
    p.ptm_low_ = 0.8f, p.ptm_low_rng_ = 0.35f, p.ptm_high_ = -0.9f, p.ptm_high_rng_ = 0.5f;
    p.ptm_high_st_ = 0.3f;
    p.brl_ = -3.0f, p.brl_r_ = 0.0f, p.brl_g_ = 0.0f, p.brl_b_ = 1.0f, p.brl_rng_ = 0.8f;
    p.brl_st_ = 0.15f;
    p.brlp_ = -1.0f, p.brlp_r_ = -1.0f, p.brlp_g_ = -1.0f, p.brlp_b_ = 0.0f;
    p.hc_r_ = 0.5f, p.hc_r_rng_ = 0.25f;
    p.hs_r_rng_ = 1.0f, p.hs_g_rng_ = 2.0f, p.hs_b_ = 0.5f, p.hs_b_rng_ = 1.5f;
    p.hs_c_ = 0.35f, p.hs_m_ = 0.25f, p.hs_y_ = 0.35f, p.hs_y_rng_ = 0.5f;
    return;
  }
  if (token == "dystopic") {
    p.tn_con_ = 1.6f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.01f, p.tn_off_ = 0.008f;
    p.tn_hcon_enable_ = 1, p.tn_hcon_ = 0.25f, p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 1.0f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f, p.tn_lcon_w_ = 0.75f;
    p.creative_white_mode_ = 3;
    p.rs_sa_ = 0.2f;
    p.pt_lml_r_ = 0.0f, p.pt_lml_g_ = 0.0f, p.pt_lml_b_ = 0.0f, p.pt_lmh_ = 0.0f;
    p.pt_lmh_r_ = 0.0f, p.pt_lmh_b_ = 0.0f;
    p.ptl_c_ = 0.05f, p.ptl_y_ = 0.05f;
    p.ptm_low_ = 0.25f, p.ptm_low_rng_ = 0.25f, p.ptm_low_st_ = 0.8f, p.ptm_high_rng_ = 0.6f;
    p.ptm_high_st_ = 0.25f;
    p.brl_ = -2.0f, p.brl_r_ = -2.0f, p.brl_g_ = -2.0f, p.brl_b_ = 0.0f, p.brl_rng_ = 0.35f;
    p.brlp_ = 0.0f, p.brlp_r_ = -1.0f, p.brlp_g_ = -1.0f, p.brlp_b_ = -1.0f;
    p.hc_r_rng_ = 0.25f;
    p.hs_r_ = 0.7f, p.hs_r_rng_ = 1.33f, p.hs_g_ = 1.0f, p.hs_g_rng_ = 2.0f, p.hs_b_ = 0.75f;
    p.hs_b_rng_ = 2.0f;
    p.hs_c_ = 1.0f, p.hs_c_rng_ = 0.5f, p.hs_m_ = 1.0f, p.hs_y_ = 1.0f, p.hs_y_rng_ = 0.765f;
    return;
  }
  if (token == "umbra") {
    p.tn_con_ = 1.8f, p.tn_sh_ = 0.5f, p.tn_toe_ = 0.001f, p.tn_off_ = 0.015f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f, p.tn_lcon_w_ = 1.0f;
    p.creative_white_mode_ = 5;
    p.pt_lml_ = 0.0f, p.pt_lml_g_ = 0.0f, p.pt_lml_b_ = 0.15f, p.pt_lmh_r_ = 0.25f;
    p.ptl_c_ = 0.05f, p.ptl_m_ = 0.06f, p.ptl_y_ = 0.05f;
    p.ptm_low_ = 0.4f, p.ptm_low_rng_ = 0.35f, p.ptm_low_st_ = 0.66f, p.ptm_high_ = -0.6f;
    p.ptm_high_rng_ = 0.45f, p.ptm_high_st_ = 0.45f;
    p.brl_ = -2.0f, p.brl_r_ = -4.5f, p.brl_g_ = -3.0f, p.brl_b_ = -4.0f, p.brl_rng_ = 0.35f;
    p.brl_st_ = 0.3f;
    p.brlp_ = 0.0f, p.brlp_r_ = -2.0f, p.brlp_g_ = -1.0f, p.brlp_b_ = -0.5f;
    p.hc_r_rng_ = 0.35f;
    p.hs_r_ = 0.66f, p.hs_g_ = 0.5f, p.hs_g_rng_ = 2.0f, p.hs_b_ = 0.85f, p.hs_b_rng_ = 2.0f;
    p.hs_c_ = 0.0f, p.hs_m_ = 0.25f, p.hs_y_ = 0.66f, p.hs_y_rng_ = 0.66f;
  }
}

void ApplyTonescalePreset(const std::string& preset, ColorUtils::OpenDRTResolvedParams& p) {
  const std::string token = NormalizeTonescalePreset(preset);
  if (token == "use_look_preset") return;
  if (token == "low_contrast") {
    p.tn_con_ = 1.4f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.005f, p.tn_lcon_enable_ = 0, p.tn_lcon_ = 0.0f;
    return;
  }
  if (token == "medium_contrast") {
    p.tn_con_ = 1.66f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.005f, p.tn_lcon_enable_ = 0, p.tn_lcon_ = 0.0f;
    return;
  }
  if (token == "high_contrast") {
    p.tn_con_ = 1.4f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.005f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f;
    return;
  }
  if (token == "arriba_tonescale") {
    p.tn_con_ = 1.05f, p.tn_toe_ = 0.1f, p.tn_off_ = 0.01f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.5f;
    p.tn_lcon_w_ = 0.2f;
    return;
  }
  if (token == "sylvan_tonescale") {
    p.tn_con_ = 1.6f, p.tn_toe_ = 0.01f, p.tn_off_ = 0.01f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.25f;
    p.tn_lcon_w_ = 0.75f;
    return;
  }
  if (token == "colorful_tonescale") {
    p.tn_con_ = 1.5f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.003f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.4f;
    p.tn_lcon_w_ = 0.5f;
    return;
  }
  if (token == "aery_tonescale") {
    p.tn_con_ = 1.15f, p.tn_toe_ = 0.04f, p.tn_off_ = 0.006f, p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 0.5f;
    p.tn_lcon_enable_ = 1, p.tn_lcon_ = 0.5f, p.tn_lcon_w_ = 2.0f;
    return;
  }
  if (token == "dystopic_tonescale") {
    p.tn_con_ = 1.6f, p.tn_toe_ = 0.01f, p.tn_off_ = 0.008f, p.tn_hcon_enable_ = 1, p.tn_hcon_ = 0.25f;
    p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 1.0f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f;
    p.tn_lcon_w_ = 0.75f;
    return;
  }
  if (token == "umbra_tonescale") {
    p.tn_con_ = 1.8f, p.tn_toe_ = 0.001f, p.tn_off_ = 0.015f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f;
    p.tn_lcon_w_ = 1.0f;
    return;
  }
  if (token == "aces_1_x") {
    p.tn_con_ = 1.0f, p.tn_sh_ = 0.35f, p.tn_toe_ = 0.02f, p.tn_off_ = 0.0f, p.tn_hcon_enable_ = 1;
    p.tn_hcon_ = 0.55f, p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 2.0f, p.tn_lcon_enable_ = 1;
    p.tn_lcon_ = 1.13f, p.tn_lcon_w_ = 1.0f;
    return;
  }
  if (token == "aces_2_0") {
    p.tn_con_ = 1.15f, p.tn_toe_ = 0.04f, p.tn_off_ = 0.0f, p.tn_hcon_enable_ = 0, p.tn_hcon_ = 1.0f;
    p.tn_hcon_pv_ = 1.0f, p.tn_hcon_st_ = 1.0f, p.tn_lcon_enable_ = 0, p.tn_lcon_ = 1.0f;
    p.tn_lcon_w_ = 0.6f;
    return;
  }
  if (token == "marvelous_tonescape") {
    p.tn_con_ = 1.5f, p.tn_toe_ = 0.003f, p.tn_off_ = 0.01f, p.tn_hcon_enable_ = 1, p.tn_hcon_ = 0.25f;
    p.tn_hcon_pv_ = 0.0f, p.tn_hcon_st_ = 4.0f, p.tn_lcon_enable_ = 1, p.tn_lcon_ = 1.0f;
    p.tn_lcon_w_ = 1.0f;
    return;
  }
  if (token == "dagrinchi_tonegroan") {
    p.tn_con_ = 1.2f, p.tn_toe_ = 0.02f, p.tn_off_ = 0.0f, p.tn_hcon_enable_ = 0, p.tn_hcon_ = 0.0f;
    p.tn_hcon_pv_ = 1.0f, p.tn_lcon_enable_ = 0, p.tn_lcon_ = 0.0f, p.tn_lcon_w_ = 0.6f;
  }
}

void ApplyCreativeWhitePreset(const std::string& preset, ColorUtils::OpenDRTResolvedParams& p) {
  const std::string token = NormalizeCreativeWhitePreset(preset);
  if (token == "use_look_preset") return;
  if (token == "d93") p.creative_white_mode_ = 0;
  if (token == "d75") p.creative_white_mode_ = 1;
  if (token == "d65") p.creative_white_mode_ = 2;
  if (token == "d60") p.creative_white_mode_ = 3;
  if (token == "d55") p.creative_white_mode_ = 4;
  if (token == "d50") p.creative_white_mode_ = 5;
}

auto init_JMhParams(const ColorUtils::ColorSpacePrimaries& prims) -> ColorUtils::JMhParams {
  using namespace ColorUtils;
  const cv::Matx33f RGB_to_XYZ = RGB_TO_XYZ_f33(prims, 1.f);
  const cv::Matx13f XYZ_w      = cv::Matx13f(ref_lum, ref_lum, ref_lum) * RGB_to_XYZ;
  const float       Y_w        = XYZ_w(1);
  const cv::Matx13f RGW_w      = XYZ_w * MATRIX_16;
  const float       k          = 1.f / (5.f * L_A + 1.f);
  const float       k4         = k * k * k * k;
  const float       F_L =
      0.2f * k4 * (5.f * L_A) + 0.1f * powf(1.f - k4, 2.f) * powf(5.f * L_A, 1.f / 3.f);
  const float       F_L_n      = F_L / ref_lum;
  const float       cz         = model_gamma;
  const cv::Matx13f D_RGB      = {F_L_n * Y_w / RGW_w(0), F_L_n * Y_w / RGW_w(1), F_L_n * Y_w / RGW_w(2)};
  const cv::Matx13f RGB_wc     = {D_RGB(0) * RGW_w(0), D_RGB(1) * RGW_w(1), D_RGB(2) * RGW_w(2)};
  const cv::Matx13f RGB_Aw     = {pacrc_fwd(RGB_wc(0)), pacrc_fwd(RGB_wc(1)), pacrc_fwd(RGB_wc(2))};
  cv::Matx33f       cone_response_to_Aab =
      cv::Matx33f::diag({cam_nl_scale, cam_nl_scale, cam_nl_scale}) * base_cone_repponse_to_Aab;
  const float A_w = cone_response_to_Aab(0, 0) * RGB_Aw(0) +
                    cone_response_to_Aab(1, 0) * RGB_Aw(1) +
                    cone_response_to_Aab(2, 0) * RGB_Aw(2);
  const float       A_w_J               = _pacrc_fwd_(F_L);
  const cv::Matx33f M1                  = RGB_to_XYZ * MATRIX_16;
  const cv::Matx33f M2                  = cv::Matx33f::diag({ref_lum, ref_lum, ref_lum});
  const cv::Matx33f MATRIX_RGB_to_CAM16 = M1 * M2;
  const cv::Matx33f MATRIX_RGB_to_CAM16_c =
      MATRIX_RGB_to_CAM16 * cv::Matx33f::diag({D_RGB(0), D_RGB(1), D_RGB(2)});
  const cv::Matx33f MATRIX_cone_response_to_Aab = {
      cone_response_to_Aab(0, 0) / A_w, cone_response_to_Aab(0, 1) * 43.f * surround[2],
      cone_response_to_Aab(0, 2) * 43.f * surround[2], cone_response_to_Aab(1, 0) / A_w,
      cone_response_to_Aab(1, 1) * 43.f * surround[2], cone_response_to_Aab(1, 2) * 43.f * surround[2],
      cone_response_to_Aab(2, 0) / A_w, cone_response_to_Aab(2, 1) * 43.f * surround[2],
      cone_response_to_Aab(2, 2) * 43.f * surround[2],
  };
  ColorUtils::JMhParams p;
  p.MATRIX_RGB_to_CAM16_c_       = MATRIX_RGB_to_CAM16_c;
  p.MATRIX_CAM16_c_to_RGB_       = MATRIX_RGB_to_CAM16_c.inv();
  p.MATRIX_cone_response_to_Aab_ = MATRIX_cone_response_to_Aab;
  p.MATRIX_Aab_to_cone_response_ = MATRIX_cone_response_to_Aab.inv();
  p.F_L_n_                       = F_L_n;
  p.cz_                          = cz;
  p.inv_cz_                      = 1.f / cz;
  p.A_w_z_                       = A_w_J;
  p.inv_A_w_J_                   = 1.f / A_w_J;
  return p;
}

}  // namespace

OutputTransformOp::OutputTransformOp() { SetParams(nlohmann::json::object()); }

OutputTransformOp::OutputTransformOp(const nlohmann::json& params) { SetParams(params); }

ColorSpace OutputTransformOp::ParseColorSpace(const std::string& cs_str) {
  const std::string token = CanonicalizeToken(cs_str);
  if (token == "rec2020") return ColorSpace::REC2020;
  if (token == "p3_d65") return ColorSpace::P3_D65;
  if (token == "p3_d60") return ColorSpace::P3_D60;
  if (token == "p3_dci") return ColorSpace::P3_DCI;
  if (token == "xyz") return ColorSpace::XYZ;
  if (token == "prophoto") return ColorSpace::PROPHOTO;
  if (token == "adobe_rgb") return ColorSpace::ADOBE_RGB;
  if (token == "ap0") return ColorSpace::AP0;
  if (token == "ap1") return ColorSpace::AP1;
  return ColorSpace::REC709;
}

ETOF OutputTransformOp::ParseETOF(const std::string& etof_str) {
  const std::string token = CanonicalizeToken(etof_str);
  if (token == "linear") return ETOF::LINEAR;
  if (token == "st2084") return ETOF::ST2084;
  if (token == "hlg") return ETOF::HLG;
  if (token == "gamma_2_6") return ETOF::GAMMA_2_6;
  if (token == "bt1886" || token == "rec1886") return ETOF::BT1886;
  if (token == "gamma_1_8") return ETOF::GAMMA_1_8;
  return ETOF::GAMMA_2_2;
}

OutputTransformMethod OutputTransformOp::ParseMethod(const std::string& method_str) {
  return CanonicalizeToken(method_str) == "aces2" ? OutputTransformMethod::ACES2
                                                   : OutputTransformMethod::OPEN_DRT;
}

std::string OutputTransformOp::ColorSpaceToString(ColorSpace cs) {
  switch (cs) {
    case ColorSpace::REC2020:
      return "rec2020";
    case ColorSpace::P3_D65:
      return "p3_d65";
    case ColorSpace::P3_D60:
      return "p3_d60";
    case ColorSpace::P3_DCI:
      return "p3_dci";
    case ColorSpace::XYZ:
      return "xyz";
    case ColorSpace::PROPHOTO:
      return "prophoto";
    case ColorSpace::ADOBE_RGB:
      return "adobe_rgb";
    case ColorSpace::AP0:
      return "ap0";
    case ColorSpace::AP1:
      return "ap1";
    default:
      return "rec709";
  }
}

std::string OutputTransformOp::ETOFToString(ETOF etof) {
  switch (etof) {
    case ETOF::LINEAR:
      return "linear";
    case ETOF::ST2084:
      return "st2084";
    case ETOF::HLG:
      return "hlg";
    case ETOF::GAMMA_2_6:
      return "gamma_2_6";
    case ETOF::BT1886:
      return "bt1886";
    case ETOF::GAMMA_1_8:
      return "gamma_1_8";
    default:
      return "gamma_2_2";
  }
}

std::string OutputTransformOp::MethodToString(OutputTransformMethod method) {
  return method == OutputTransformMethod::ACES2 ? "aces2" : "open_drt";
}

void OutputTransformOp::Apply(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error(
      "OutputTransformOp: Use the pipeline output-transform stage. This operator is only a "
      "descriptor for the ODT stage.");
}

void OutputTransformOp::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("OutputTransformOp: GPU Apply is not supported directly.");
}

auto OutputTransformOp::GetParams() const -> nlohmann::json {
  return {{std::string(script_name_), authoring_params_}};
}

void OutputTransformOp::SetParams(const nlohmann::json& in_j) {
  authoring_params_ = DefaultAuthoringParams();

  const nlohmann::json* incoming = nullptr;
  if (in_j.contains(script_name_)) {
    incoming = &in_j.at(script_name_);
  }
  if (incoming && incoming->is_object()) {
    MergeJson(authoring_params_, *incoming);
  }

  authoring_params_["method"] = MethodToString(ParseMethod(authoring_params_.value("method", std::string("open_drt"))));
  authoring_params_["encoding_space"] =
      ColorSpaceToString(ParseColorSpace(authoring_params_.value("encoding_space", std::string("rec709"))));
  authoring_params_["encoding_etof"] =
      ETOFToString(ParseETOF(authoring_params_.value("encoding_etof", std::string("gamma_2_2"))));
  authoring_params_["limiting_space"] =
      ColorSpaceToString(ParseColorSpace(authoring_params_.value("limiting_space", std::string("rec709"))));
  if (!authoring_params_.contains("peak_luminance") || !authoring_params_["peak_luminance"].is_number()) {
    authoring_params_["peak_luminance"] = 100.0f;
  }

  if (!authoring_params_.contains("open_drt") || !authoring_params_["open_drt"].is_object()) {
    authoring_params_["open_drt"] = DefaultOpenDRTAuthoringParams();
  } else {
    nlohmann::json merged = DefaultOpenDRTAuthoringParams();
    MergeJson(merged, authoring_params_["open_drt"]);
    authoring_params_["open_drt"] = std::move(merged);
  }

  auto& open_drt_json = authoring_params_["open_drt"];
  open_drt_json["look_preset"] =
      NormalizeLookPreset(open_drt_json.value("look_preset", std::string("standard")));
  open_drt_json["tonescale_preset"] =
      NormalizeTonescalePreset(open_drt_json.value("tonescale_preset", std::string("use_look_preset")));
  open_drt_json["display_encoding_preset"] =
      NormalizeDisplayEncodingPreset(open_drt_json.value("display_encoding_preset", std::string("srgb_display")));
  open_drt_json["creative_white_preset"] =
      NormalizeCreativeWhitePreset(open_drt_json.value("creative_white_preset", std::string("use_look_preset")));

  if (incoming && incoming->is_object() && incoming->contains("open_drt") &&
      (*incoming)["open_drt"].is_object() &&
      (*incoming)["open_drt"].contains("display_encoding_preset")) {
    const bool overrides_common =
        incoming->contains("encoding_space") || incoming->contains("encoding_etof") ||
        incoming->contains("limiting_space");
    if (!overrides_common) {
      auto [encoding_space, encoding_etof, limiting_space] =
          DisplayPresetToCommonFields(open_drt_json["display_encoding_preset"].get<std::string>());
      authoring_params_["encoding_space"] = ColorSpaceToString(encoding_space);
      authoring_params_["encoding_etof"]  = ETOFToString(encoding_etof);
      authoring_params_["limiting_space"] = ColorSpaceToString(limiting_space);
    }
  }

  ResolveOutputTransform();
}

void OutputTransformOp::ResolveOutputTransform() {
  const ColorSpace encoding_space =
      ParseColorSpace(authoring_params_.value("encoding_space", std::string("rec709")));
  const ColorSpace limiting_space =
      ParseColorSpace(authoring_params_.value("limiting_space", std::string("rec709")));
  const ETOF encoding_etof =
      ParseETOF(authoring_params_.value("encoding_etof", std::string("gamma_2_2")));

  to_output_params_                      = {};
  to_output_params_.method_              =
      ParseMethod(authoring_params_.value("method", std::string("open_drt")));
  to_output_params_.limit_to_display_matx_ =
      ColorUtils::RGB_TO_XYZ_f33(limiting_space) * ColorUtils::XYZ_TO_RGB_f33(encoding_space);
  to_output_params_.etof_                 = encoding_etof;
  to_output_params_.display_linear_scale_ = DetermineDisplayLinearScale(encoding_etof);

  if (to_output_params_.method_ == OutputTransformMethod::ACES2) {
    ResolveACESParams();
  } else {
    ResolveOpenDRTParams();
  }
}

void OutputTransformOp::ResolveACESParams() {
  const ColorSpace encoding_space =
      ParseColorSpace(authoring_params_.value("encoding_space", std::string("rec709")));
  const ColorSpace limiting_space =
      ResolveACESLimitingSpace(ParseColorSpace(authoring_params_.value("limiting_space", std::string("rec709"))));
  const float peak_luminance = authoring_params_.value("peak_luminance", 100.0f);

  const ODTCacheKey cache_key = BuildODTCacheKey(encoding_space, limiting_space, peak_luminance);
  std::lock_guard<std::mutex> lock(ODTParamsCacheMutex());
  auto&                       cache = ODTParamsCache();
  if (const auto it = cache.find(cache_key); it != cache.end()) {
    to_output_params_.aces_odt_params_ = it->second;
    return;
  }

  using namespace ColorUtils;
  auto& odt                 = to_output_params_.aces_odt_params_;
  odt.peak_luminance_       = peak_luminance;
  odt.input_params_         = ::puerhlab::init_JMhParams(AP0_PRIMARY);
  odt.reach_params_         = ::puerhlab::init_JMhParams(REACH_PRIMARY);
  odt.limit_params_         = ::puerhlab::init_JMhParams(SpaceEnumToPrimary(limiting_space));

  const float n             = peak_luminance;
  const float n_r           = 100.0f;
  const float g             = 1.15f;
  const float c             = 0.18f;
  const float c_d           = 10.013f;
  const float w_g           = 0.14f;
  const float t_1           = 0.04f;
  const float r_hit_min     = 128.f;
  const float r_hit_max     = 896.f;
  const float r_hit         = r_hit_min + (r_hit_max - r_hit_min) * (log(n / n_r) / log(10000.f / 100.f));
  const float m_0           = (n / n_r);
  const float m_1           = 0.5f * (m_0 + sqrt(m_0 * (m_0 + 4.f * t_1)));
  const float u             = pow((r_hit / m_1) / ((r_hit / m_1) + 1), g);
  const float m             = m_1 / u;
  const float w_i           = log(n / 100.f) / log(2.f);
  const float c_t           = c_d / n_r * (1.f + w_i * w_g);
  const float g_ip          = 0.5f * (c_t + sqrt(c_t * (c_t + 4.f * t_1)));
  const float g_ipp2        = -(m_1 * pow((g_ip / m), (1.f / g))) / (pow(g_ip / m, 1.f / g) - 1.f);
  const float w_2           = c / g_ipp2;
  const float s_2           = w_2 * m_1;
  const float u_2           = pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
  const float m_2           = m_1 / u_2;

  odt.ts_params_ = {
      n, n_r, g, t_1, c_t, s_2, u_2, m_2, 8.f * r_hit, n / (u_2 * n_r), log10(n / n_r)};

  odt.limit_J_max_          = Y_to_J(peak_luminance, odt.input_params_);
  odt.model_gamma_inv_      = 1.f / model_gamma;
  odt.table_reach_M_        = MakeReachMTable(odt.reach_params_, odt.limit_J_max_);
  odt.sat_                  = fmaxf(0.2f, chroma_expand - (chroma_expand * chroma_expand_fact) * odt.ts_params_.log_peak_);
  odt.sat_thr_              = chroma_expand_thr / peak_luminance;
  odt.compr_                = chroma_compress + (chroma_compress * chroma_compress_fact) * odt.ts_params_.log_peak_;
  odt.chroma_compress_scale_ = powf(0.03379f * peak_luminance, 0.30596f) - 0.45135f;
  odt.mid_J_                = Y_to_J(odt.ts_params_.c_t_ * ref_lum, odt.input_params_);
  odt.focus_dist_           = focus_distance + focus_distance * focus_distance_scaling * odt.ts_params_.log_peak_;
  const float lower_gamma   = 1.14f + 0.07f * odt.ts_params_.log_peak_;
  odt.lower_hull_gamma_     = lower_gamma;
  odt.lower_hull_gamma_inv_ = 1.f / lower_gamma;
  odt.table_gamut_cusps_    = MakeUniformHueGamutTable(odt.reach_params_, odt.limit_params_, odt);
  odt.table_hues_           = std::make_shared<std::array<float, TOTAL_TABLE_SIZE>>();
  for (int i = 0; i < TOTAL_TABLE_SIZE; ++i) {
    (*odt.table_hues_)[i] = (*odt.table_gamut_cusps_)[i](2);
  }
  odt.table_upper_hull_gammas_ =
      std::make_shared<std::array<float, TOTAL_TABLE_SIZE>>(MakeUpperHullGammaTable(*odt.table_gamut_cusps_, odt));
  odt.hue_linearity_search_range_ = DetermineHueLinearitySearchRange(*odt.table_hues_);

  cache.emplace(cache_key, odt);
}

void OutputTransformOp::ResolveOpenDRTParams() {
  const auto& open_drt_json  = authoring_params_.at("open_drt");
  const ColorSpace encoding_space =
      ParseColorSpace(authoring_params_.value("encoding_space", std::string("rec709")));
  const ColorSpace limiting_space =
      ParseColorSpace(authoring_params_.value("limiting_space", std::string("rec709")));
  const ETOF encoding_etof =
      ParseETOF(authoring_params_.value("encoding_etof", std::string("gamma_2_2")));

  ColorUtils::OpenDRTResolvedParams resolved;
  ApplyLookPreset(open_drt_json.value("look_preset", std::string("standard")), resolved);
  ApplyTonescalePreset(open_drt_json.value("tonescale_preset", std::string("use_look_preset")), resolved);
  ApplyCreativeWhitePreset(
      open_drt_json.value("creative_white_preset", std::string("use_look_preset")), resolved);

  resolved.peak_luminance_ = authoring_params_.value("peak_luminance", 100.0f);
  OverrideFloat(open_drt_json, "tn_Lp", resolved.peak_luminance_);
  OverrideFloat(open_drt_json, "tn_gb", resolved.tn_gb_);
  OverrideFloat(open_drt_json, "pt_hdr", resolved.pt_hdr_);
  OverrideFloat(open_drt_json, "tn_Lg", resolved.tn_Lg_);

  OverrideFloat(open_drt_json, "tn_con", resolved.tn_con_);
  OverrideFloat(open_drt_json, "tn_sh", resolved.tn_sh_);
  OverrideFloat(open_drt_json, "tn_toe", resolved.tn_toe_);
  OverrideFloat(open_drt_json, "tn_off", resolved.tn_off_);
  OverrideBoolLike(open_drt_json, "tn_hcon_enable", resolved.tn_hcon_enable_);
  OverrideFloat(open_drt_json, "tn_hcon", resolved.tn_hcon_);
  OverrideFloat(open_drt_json, "tn_hcon_pv", resolved.tn_hcon_pv_);
  OverrideFloat(open_drt_json, "tn_hcon_st", resolved.tn_hcon_st_);
  OverrideBoolLike(open_drt_json, "tn_lcon_enable", resolved.tn_lcon_enable_);
  OverrideFloat(open_drt_json, "tn_lcon", resolved.tn_lcon_);
  OverrideFloat(open_drt_json, "tn_lcon_w", resolved.tn_lcon_w_);
  OverrideFloat(open_drt_json, "rs_sa", resolved.rs_sa_);
  OverrideFloat(open_drt_json, "rs_rw", resolved.rs_rw_);
  OverrideFloat(open_drt_json, "rs_bw", resolved.rs_bw_);
  OverrideBoolLike(open_drt_json, "pt_enable", resolved.pt_enable_);
  OverrideFloat(open_drt_json, "pt_lml", resolved.pt_lml_);
  OverrideFloat(open_drt_json, "pt_lml_r", resolved.pt_lml_r_);
  OverrideFloat(open_drt_json, "pt_lml_g", resolved.pt_lml_g_);
  OverrideFloat(open_drt_json, "pt_lml_b", resolved.pt_lml_b_);
  OverrideFloat(open_drt_json, "pt_lmh", resolved.pt_lmh_);
  OverrideFloat(open_drt_json, "pt_lmh_r", resolved.pt_lmh_r_);
  OverrideFloat(open_drt_json, "pt_lmh_b", resolved.pt_lmh_b_);
  OverrideBoolLike(open_drt_json, "ptl_enable", resolved.ptl_enable_);
  OverrideFloat(open_drt_json, "ptl_c", resolved.ptl_c_);
  OverrideFloat(open_drt_json, "ptl_m", resolved.ptl_m_);
  OverrideFloat(open_drt_json, "ptl_y", resolved.ptl_y_);
  OverrideBoolLike(open_drt_json, "ptm_enable", resolved.ptm_enable_);
  OverrideFloat(open_drt_json, "ptm_low", resolved.ptm_low_);
  OverrideFloat(open_drt_json, "ptm_low_rng", resolved.ptm_low_rng_);
  OverrideFloat(open_drt_json, "ptm_low_st", resolved.ptm_low_st_);
  OverrideFloat(open_drt_json, "ptm_high", resolved.ptm_high_);
  OverrideFloat(open_drt_json, "ptm_high_rng", resolved.ptm_high_rng_);
  OverrideFloat(open_drt_json, "ptm_high_st", resolved.ptm_high_st_);
  OverrideBoolLike(open_drt_json, "brl_enable", resolved.brl_enable_);
  OverrideFloat(open_drt_json, "brl", resolved.brl_);
  OverrideFloat(open_drt_json, "brl_r", resolved.brl_r_);
  OverrideFloat(open_drt_json, "brl_g", resolved.brl_g_);
  OverrideFloat(open_drt_json, "brl_b", resolved.brl_b_);
  OverrideFloat(open_drt_json, "brl_rng", resolved.brl_rng_);
  OverrideFloat(open_drt_json, "brl_st", resolved.brl_st_);
  OverrideBoolLike(open_drt_json, "brlp_enable", resolved.brlp_enable_);
  OverrideFloat(open_drt_json, "brlp", resolved.brlp_);
  OverrideFloat(open_drt_json, "brlp_r", resolved.brlp_r_);
  OverrideFloat(open_drt_json, "brlp_g", resolved.brlp_g_);
  OverrideFloat(open_drt_json, "brlp_b", resolved.brlp_b_);
  OverrideBoolLike(open_drt_json, "hc_enable", resolved.hc_enable_);
  OverrideFloat(open_drt_json, "hc_r", resolved.hc_r_);
  OverrideFloat(open_drt_json, "hc_r_rng", resolved.hc_r_rng_);
  OverrideBoolLike(open_drt_json, "hs_rgb_enable", resolved.hs_rgb_enable_);
  OverrideFloat(open_drt_json, "hs_r", resolved.hs_r_);
  OverrideFloat(open_drt_json, "hs_r_rng", resolved.hs_r_rng_);
  OverrideFloat(open_drt_json, "hs_g", resolved.hs_g_);
  OverrideFloat(open_drt_json, "hs_g_rng", resolved.hs_g_rng_);
  OverrideFloat(open_drt_json, "hs_b", resolved.hs_b_);
  OverrideFloat(open_drt_json, "hs_b_rng", resolved.hs_b_rng_);
  OverrideBoolLike(open_drt_json, "hs_cmy_enable", resolved.hs_cmy_enable_);
  OverrideFloat(open_drt_json, "hs_c", resolved.hs_c_);
  OverrideFloat(open_drt_json, "hs_c_rng", resolved.hs_c_rng_);
  OverrideFloat(open_drt_json, "hs_m", resolved.hs_m_);
  OverrideFloat(open_drt_json, "hs_m_rng", resolved.hs_m_rng_);
  OverrideFloat(open_drt_json, "hs_y", resolved.hs_y_);
  OverrideFloat(open_drt_json, "hs_y_rng", resolved.hs_y_rng_);
  OverrideFloat(open_drt_json, "cwp_lm", resolved.creative_white_luminance_mix_);
  if (open_drt_json.contains("creative_white_mode") && open_drt_json["creative_white_mode"].is_number_integer()) {
    resolved.creative_white_mode_ = std::clamp(open_drt_json["creative_white_mode"].get<int>(), 0, 5);
  }

  resolved.rs_w_[0] = resolved.rs_rw_;
  resolved.rs_w_[2] = resolved.rs_bw_;
  resolved.rs_w_[1] = 1.0f - resolved.rs_rw_ - resolved.rs_bw_;

  const int surround_comp = DetermineOpenDRTSurroundComp(encoding_space, encoding_etof);
  resolved.ts_x1_         = std::pow(2.0f, 6.0f * resolved.tn_sh_ + 4.0f);
  resolved.ts_y1_         = resolved.peak_luminance_ / 100.0f;
  resolved.ts_x0_         = 0.18f + resolved.tn_off_;
  resolved.ts_y0_         = resolved.tn_Lg_ / 100.0f * (1.0f + resolved.tn_gb_ * std::log2(resolved.ts_y1_));
  resolved.ts_s0_         = CompressToeQuadratic(resolved.ts_y0_, resolved.tn_toe_, true);
  resolved.ts_p_          = resolved.tn_con_ / (1.0f + static_cast<float>(surround_comp) * 0.05f);
  resolved.ts_s10_        = resolved.ts_x0_ * (std::pow(resolved.ts_s0_, -1.0f / resolved.tn_con_) - 1.0f);
  resolved.ts_m1_         = resolved.ts_y1_ / std::pow(resolved.ts_x1_ / (resolved.ts_x1_ + resolved.ts_s10_), resolved.tn_con_);
  resolved.ts_m2_         = CompressToeQuadratic(resolved.ts_m1_, resolved.tn_toe_, true);
  resolved.ts_s_          =
      resolved.ts_x0_ * (std::pow(resolved.ts_s0_ / resolved.ts_m2_, -1.0f / resolved.tn_con_) - 1.0f);
  resolved.ts_dsc_        = encoding_etof == ETOF::ST2084
                                ? 0.01f
                                : (encoding_etof == ETOF::HLG ? 0.1f : 100.0f / resolved.peak_luminance_);
  resolved.pt_cmp_Lf_     =
      resolved.pt_hdr_ * std::min(1.0f, (resolved.peak_luminance_ - 100.0f) / 900.0f);
  resolved.s_Lp100_       =
      resolved.ts_x0_ * (std::pow(resolved.tn_Lg_ / 100.0f, -1.0f / resolved.tn_con_) - 1.0f);
  resolved.ts_s1_         = resolved.ts_s_ * resolved.pt_cmp_Lf_ +
                    resolved.s_Lp100_ * (1.0f - resolved.pt_cmp_Lf_);
  resolved.output_linear_scale_ = DetermineDisplayLinearScale(encoding_etof);

  resolved.ap1_to_render_mat_ =
      ColorUtils::RGB_TO_XYZ_f33(ColorSpace::AP1) * ColorUtils::XYZ_TO_RGB_f33(ColorSpace::P3_D65);
  resolved.render_to_limit_neutral_mat_ =
      MakeRenderToLimitMatrix(limiting_space, LimitingSpaceWhiteXY(limiting_space));
  resolved.render_to_limit_creative_mat_ =
      MakeRenderToLimitMatrix(limiting_space, WhitePointXYFromMode(resolved.creative_white_mode_));
  resolved.creative_white_norm_ =
      WhiteNormFromRenderToLimit(resolved.render_to_limit_creative_mat_);

  to_output_params_.open_drt_params_      = resolved;
  to_output_params_.display_linear_scale_ = resolved.output_linear_scale_;
}

void OutputTransformOp::SetGlobalParams(OperatorParams& global_params) const {
  global_params.to_output_params_ = to_output_params_;
  global_params.to_output_dirty_  = true;
}

void OutputTransformOp::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.to_output_enabled_ = enable;
  params.to_output_dirty_   = true;
}
}  // namespace puerhlab
