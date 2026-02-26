#pragma once

#include "json.hpp"

namespace puerhlab::pipeline_defaults {

inline auto MakeDefaultRawDecodeParams() -> nlohmann::json {
  nlohmann::json decode_params;
#ifdef HAVE_CUDA
  decode_params["raw"]["cuda"] = true;
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
           {{"enabled", true},
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

}  // namespace puerhlab::pipeline_defaults
