//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "json.hpp"

namespace puerhlab::pipeline_defaults {

inline auto MakeDefaultRawDecodeParams() -> nlohmann::json {
  nlohmann::json decode_params;
  decode_params["raw"]["gpu_backend"] = "cpu";
#ifdef HAVE_CUDA
  decode_params["raw"]["gpu_backend"] = "cuda";
  decode_params["raw"]["cuda"]        = true;
#else
  decode_params["raw"]["cuda"] = false;
#endif
  decode_params["raw"]["highlights_reconstruct"] = true;
  decode_params["raw"]["use_camera_wb"]          = true;
  decode_params["raw"]["user_wb"]                = 7600.0f;
  decode_params["raw"]["backend"]                = "puerh";
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
              {"hdr_purity", 0.5f}}}}}};
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

}  // namespace puerhlab::pipeline_defaults
