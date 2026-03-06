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
#include <memory>
#include <stdexcept>

#include "edit/operators/cst/aces_odt_cpu.hpp"

namespace puerhlab {

ODT_Op::ODT_Op(const nlohmann::json& params) { SetParams(params); }

void ODT_Op::Apply(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error(
      "ODT_Op: Use the pipeline output transform stage. This operator is a descriptor for the "
      "GPU/CPU output transform runtime.");
}

void ODT_Op::ApplyGPU(std::shared_ptr<ImageBuffer>) {
  throw std::runtime_error("ODT_Op: GPU Apply is handled by the pipeline output stage.");
}

auto ODT_Op::GetParams() const -> nlohmann::json {
  nlohmann::json odt;
  odt["method"]         = ColorUtils::ODTMethodToString(method_);
  odt["encoding_space"] = ColorUtils::ColorSpaceToString(encoding_space_);
  odt["encoding_eotf"]  = ColorUtils::EOTFToString(encoding_eotf_);
  odt["limiting_space"] = ColorUtils::ColorSpaceToString(limiting_space_);
  odt["peak_luminance"] = peak_luminance_;

  odt["open_drt"] = {
      {"look_preset", odt_cpu::OpenDRTLookPresetToString(open_drt_settings_.look_preset_)},
      {"tonescale_preset",
       odt_cpu::OpenDRTTonescalePresetToString(open_drt_settings_.tonescale_preset_)},
      {"creative_white",
       odt_cpu::OpenDRTCreativeWhitePresetToString(open_drt_settings_.creative_white_)},
      {"creative_white_limit", open_drt_settings_.creative_white_limit_},
      {"display_grey_luminance", open_drt_settings_.display_grey_luminance_},
      {"hdr_grey_boost", open_drt_settings_.hdr_grey_boost_},
      {"hdr_purity", open_drt_settings_.hdr_purity_},
  };

  return {{std::string(script_name_), odt}};
}

void ODT_Op::SetParams(const nlohmann::json& in_j) {
  if (!in_j.contains(script_name_) || !in_j.at(script_name_).is_object()) {
    return;
  }

  const auto& j = in_j.at(script_name_);

  if (j.contains("method") && j.at("method").is_string()) {
    const std::string method_str = j.at("method").get<std::string>();
    if (method_str != "open_drt" && method_str != "aces_2_0") {
      throw std::runtime_error("ODT_Op: unsupported odt.method \"" + method_str + "\".");
    }
    method_ = ColorUtils::ODTMethodFromString(method_str);
  }
  if (j.contains("encoding_space") && j.at("encoding_space").is_string()) {
    encoding_space_ = ColorUtils::ColorSpaceFromString(j.at("encoding_space").get<std::string>());
  }
  if (j.contains("encoding_eotf") && j.at("encoding_eotf").is_string()) {
    encoding_eotf_ = ColorUtils::EOTFFromString(j.at("encoding_eotf").get<std::string>());
  }
  if (j.contains("limiting_space") && j.at("limiting_space").is_string()) {
    limiting_space_ = ColorUtils::ColorSpaceFromString(j.at("limiting_space").get<std::string>());
  }
  if (j.contains("peak_luminance") && j.at("peak_luminance").is_number()) {
    peak_luminance_ = j.at("peak_luminance").get<float>();
  }

  if (j.contains("open_drt") && j.at("open_drt").is_object()) {
    const auto& open_drt = j.at("open_drt");
    if (open_drt.contains("look_preset") && open_drt.at("look_preset").is_string()) {
      open_drt_settings_.look_preset_ =
          odt_cpu::OpenDRTLookPresetFromString(open_drt.at("look_preset").get<std::string>());
    }
    if (open_drt.contains("tonescale_preset") && open_drt.at("tonescale_preset").is_string()) {
      open_drt_settings_.tonescale_preset_ = odt_cpu::OpenDRTTonescalePresetFromString(
          open_drt.at("tonescale_preset").get<std::string>());
    }
    if (open_drt.contains("creative_white") && open_drt.at("creative_white").is_string()) {
      open_drt_settings_.creative_white_ = odt_cpu::OpenDRTCreativeWhitePresetFromString(
          open_drt.at("creative_white").get<std::string>());
    }
    if (open_drt.contains("creative_white_limit") &&
        open_drt.at("creative_white_limit").is_number()) {
      open_drt_settings_.creative_white_limit_ =
          open_drt.at("creative_white_limit").get<float>();
    }
    if (open_drt.contains("display_grey_luminance") &&
        open_drt.at("display_grey_luminance").is_number()) {
      open_drt_settings_.display_grey_luminance_ =
          open_drt.at("display_grey_luminance").get<float>();
    }
    if (open_drt.contains("hdr_grey_boost") && open_drt.at("hdr_grey_boost").is_number()) {
      open_drt_settings_.hdr_grey_boost_ = open_drt.at("hdr_grey_boost").get<float>();
    }
    if (open_drt.contains("hdr_purity") && open_drt.at("hdr_purity").is_number()) {
      open_drt_settings_.hdr_purity_ = open_drt.at("hdr_purity").get<float>();
    }
  }

  ValidateParams();
  RebuildRuntime();
}

void ODT_Op::ValidateParams() const {
  if (peak_luminance_ <= 0.0f) {
    throw std::runtime_error("ODT_Op: peak_luminance must be positive.");
  }

  if (method_ == ColorUtils::ODTMethod::OPEN_DRT) {
    if (encoding_space_ == ColorUtils::ColorSpace::PROPHOTO ||
        encoding_space_ == ColorUtils::ColorSpace::ADOBE_RGB) {
      throw std::runtime_error("ODT_Op: OpenDRT does not support encoding_space \"" +
                               ColorUtils::ColorSpaceToString(encoding_space_) + "\".");
    }
  } else {
    if (limiting_space_ == ColorUtils::ColorSpace::XYZ) {
      throw std::runtime_error("ODT_Op: ACES 2.0 limiting_space does not support \"xyz\".");
    }
  }
}

void ODT_Op::RebuildRuntime() {
  to_output_params_                 = {};
  to_output_params_.method_         = method_;
  to_output_params_.encoding_space_ = encoding_space_;
  to_output_params_.eotf_           = encoding_eotf_;
  to_output_params_.peak_luminance_ = peak_luminance_;

  if (method_ == ColorUtils::ODTMethod::ACES_2_0) {
    to_output_params_.aces_params_ =
        odt_cpu::ResolveACESODTRuntime(limiting_space_, peak_luminance_);
    to_output_params_.limit_to_display_matx_ =
        ColorUtils::RGB_TO_XYZ_f33(limiting_space_) * ColorUtils::XYZ_TO_RGB_f33(encoding_space_);
    to_output_params_.display_linear_scale_ = 1.0f;
    return;
  }

  to_output_params_.open_drt_params_ = odt_cpu::ResolveOpenDRTRuntime(
      encoding_space_, encoding_eotf_, peak_luminance_, open_drt_settings_);
  to_output_params_.limit_to_display_matx_ = cv::Matx33f::eye();
  to_output_params_.display_linear_scale_ =
      odt_cpu::ResolveOpenDRTDisplayLinearScale(to_output_params_.open_drt_params_);
}

void ODT_Op::SetGlobalParams(OperatorParams& global_params) const {
  global_params.to_output_params_ = to_output_params_;
  global_params.to_output_dirty_  = true;
}

void ODT_Op::EnableGlobalParams(OperatorParams& params, bool enable) {
  params.to_output_enabled_ = enable;
  params.to_output_dirty_   = enable;
}

}  // namespace puerhlab
