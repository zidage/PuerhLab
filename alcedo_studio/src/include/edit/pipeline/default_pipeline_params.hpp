//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "json.hpp"

namespace alcedo::pipeline_defaults {

inline auto MakeDefaultRawDecodeParams() -> nlohmann::json {
  nlohmann::json decode_params;
  decode_params["raw"]["gpu_backend"] = "cpu";
#if defined(HAVE_CUDA) || defined(HAVE_METAL)
  decode_params["raw"]["gpu_backend"] = "gpu";
#endif
  decode_params["raw"]["cuda"] = false;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = true;
#endif
  decode_params["raw"]["highlights_reconstruct"] = true;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 7600.0f;
  decode_params["raw"]["backend"]                = "alcedo";
  return decode_params;
}

inline auto MakeDefaultLensCalibParams() -> nlohmann::json {
  return {{"lens_calib",
           {{"enabled", false},
            {"apply_vignetting", true},
            {"apply_distortion", true},
            {"apply_tca", true},
            {"apply_crop", true},
            {"auto_scale", true},
            {"use_user_scale", false},
            {"user_scale", 1.0f},
            {"projection_enabled", false},
            {"target_projection", "unknown"},
            {"lens_profile_db_path", "src/config/lens_calib"}}}};
}

inline auto MakeDefaultODTParams() -> nlohmann::json {
  return {{"odt",
           {{"method", "open_drt"},
            {"encoding_space", "rec709"},
            {"encoding_eotf", "gamma_2_2"},
            {"limiting_space", "rec709"},
            {"peak_luminance", 100.0f},
            {"open_drt",
             {{"look_preset", "standard"},
              {"tonescale_preset", "use_look_preset"},
              {"creative_white", "use_look_preset"},
              {"creative_white_limit", 0.25f},
              {"display_grey_luminance", 10.0f},
              {"hdr_grey_boost", 0.13f},
              {"hdr_purity", 0.5f},
              {"parameters", {{"tn_con", 1.66f},       {"tn_sh", 0.5f},       {"tn_toe", 0.003f},
                              {"tn_off", 0.005f},      {"tn_hcon", 0.0f},     {"tn_hcon_pv", 1.0f},
                              {"tn_hcon_st", 4.0f},    {"tn_lcon", 0.0f},     {"tn_lcon_w", 0.5f},
                              {"cwp_lm", 0.25f},       {"rs_sa", 0.35f},      {"rs_rw", 0.25f},
                              {"rs_bw", 0.55f},        {"pt_lml", 0.25f},     {"pt_lml_r", 0.5f},
                              {"pt_lml_g", 0.0f},      {"pt_lml_b", 0.1f},    {"pt_lmh", 0.25f},
                              {"pt_lmh_r", 0.5f},      {"pt_lmh_b", 0.0f},    {"ptl_c", 0.06f},
                              {"ptl_m", 0.08f},        {"ptl_y", 0.06f},      {"ptm_low", 0.4f},
                              {"ptm_low_rng", 0.25f},  {"ptm_low_st", 0.5f},  {"ptm_high", -0.8f},
                              {"ptm_high_rng", 0.35f}, {"ptm_high_st", 0.4f}, {"brl", 0.0f},
                              {"brl_r", -2.5f},        {"brl_g", -1.5f},      {"brl_b", -1.5f},
                              {"brl_rng", 0.5f},       {"brl_st", 0.35f},     {"brlp", -0.5f},
                              {"brlp_r", -1.25f},      {"brlp_g", -1.25f},    {"brlp_b", -0.25f},
                              {"hc_r", 1.0f},          {"hc_r_rng", 0.3f},    {"hs_r", 0.6f},
                              {"hs_r_rng", 0.6f},      {"hs_g", 0.35f},       {"hs_g_rng", 1.0f},
                              {"hs_b", 0.66f},         {"hs_b_rng", 1.0f},    {"hs_c", 0.25f},
                              {"hs_c_rng", 1.0f},      {"hs_m", 0.0f},        {"hs_m_rng", 1.0f},
                              {"hs_y", 0.0f},          {"hs_y_rng", 1.0f}}}}}}}};
}

inline auto MakeDefaultCropRotateParams() -> nlohmann::json {
  return {{"crop_rotate",
           {{"enabled", false},
            {"angle_degrees", 0.0f},
            {"enable_crop", true},
            {"crop_rect", {{"x", 0.0f}, {"y", 0.0f}, {"w", 1.0f}, {"h", 1.0f}}},
            {"expand_to_fit", true},
            {"aspect_ratio_preset", "free"},
            {"aspect_ratio", {{"width", 1.0f}, {"height", 1.0f}}}}}};
}

}  // namespace alcedo::pipeline_defaults
