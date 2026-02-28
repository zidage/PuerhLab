// pipeline_io.cpp — Implementations for AdjustmentField <-> pipeline mapping.
//
// Extracted from dialog.cpp.  Every function is stateless and depends only on
// its explicit arguments; none of them touch Qt widgets.

#include "ui/puerhlab_main/editor_dialog/modules/pipeline_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QPointF>
#include <json.hpp>

#include "edit/pipeline/default_pipeline_params.hpp"
#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/color_temp.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/color_wheel.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/curve.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/geometry.hpp"
#include "ui/puerhlab_main/editor_dialog/modules/hls.hpp"
#include "ui/puerhlab_main/editor_dialog/state.hpp"

namespace puerhlab::ui::pipeline_io {

// =========================================================================
// Low-level pipeline-stage readers
// =========================================================================

auto IsOperatorEnabled(const PipelineStage& stage,
                       OperatorType type) -> std::optional<bool> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (!j.contains("enable")) {
    return true;
  }
  try {
    return j["enable"].get<bool>();
  } catch (...) {
    return std::nullopt;
  }
}

auto ReadFloat(const PipelineStage& stage, OperatorType type,
               const char* key) -> std::optional<float> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (j.contains("enable") && !j["enable"].get<bool>()) {
    return std::nullopt;
  }
  if (!j.contains("params")) {
    return std::nullopt;
  }
  const auto& params = j["params"];
  if (!params.contains(key)) {
    return std::nullopt;
  }
  try {
    return params[key].get<float>();
  } catch (...) {
    return std::nullopt;
  }
}

auto ReadNestedFloat(const PipelineStage& stage, OperatorType type, const char* key1,
                     const char* key2) -> std::optional<float> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (j.contains("enable") && !j["enable"].get<bool>()) {
    return std::nullopt;
  }
  if (!j.contains("params")) {
    return std::nullopt;
  }
  const auto& params = j["params"];
  if (!params.contains(key1)) {
    return std::nullopt;
  }
  const auto& inner = params[key1];
  if (!inner.contains(key2)) {
    return std::nullopt;
  }
  try {
    return inner[key2].get<float>();
  } catch (...) {
    return std::nullopt;
  }
}

auto ReadNestedObject(const PipelineStage& stage, OperatorType type,
                      const char* key) -> std::optional<nlohmann::json> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (!j.contains("params")) {
    return std::nullopt;
  }
  const auto& params = j["params"];
  if (!params.contains(key) || !params[key].is_object()) {
    return std::nullopt;
  }
  return params[key];
}

auto ReadString(const PipelineStage& stage, OperatorType type,
                const char* key) -> std::optional<std::string> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (j.contains("enable") && !j["enable"].get<bool>()) {
    return std::nullopt;
  }
  if (!j.contains("params")) {
    return std::nullopt;
  }
  const auto& params = j["params"];
  if (!params.contains(key)) {
    return std::nullopt;
  }
  try {
    return params[key].get<std::string>();
  } catch (...) {
    return std::nullopt;
  }
}

auto ReadCurvePoints(const PipelineStage& stage,
                     OperatorType type) -> std::optional<std::vector<QPointF>> {
  const auto op = stage.GetOperator(type);
  if (!op.has_value() || op.value() == nullptr) {
    return std::nullopt;
  }
  const auto j = op.value()->ExportOperatorParams();
  if (j.contains("enable")) {
    try {
      if (!j["enable"].get<bool>()) {
        return curve::DefaultCurveControlPoints();
      }
    } catch (...) {
    }
  }
  if (!j.contains("params")) {
    return std::nullopt;
  }
  return curve::ParseCurveControlPointsFromParams(j["params"]);
}

// =========================================================================
// ReadCurrentOperatorParams
// =========================================================================

auto ReadCurrentOperatorParams(CPUPipelineExecutor& exec, PipelineStageName stage_name,
                               OperatorType op_type) -> std::optional<nlohmann::json> {
  const auto op = exec.GetStage(stage_name).GetOperator(op_type);
  if (!op.has_value() || !op.value() || !op.value()->op_) {
    return std::nullopt;
  }
  const auto exported = op.value()->ExportOperatorParams();
  if (!exported.contains("params") || !exported["params"].is_object()) {
    return std::nullopt;
  }
  return exported["params"];
}

// =========================================================================
// FieldSpec
// =========================================================================

auto FieldSpec(AdjustmentField field) -> std::pair<PipelineStageName, OperatorType> {
  switch (field) {
    case AdjustmentField::Exposure:
      return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
    case AdjustmentField::Contrast:
      return {PipelineStageName::Basic_Adjustment, OperatorType::CONTRAST};
    case AdjustmentField::Saturation:
      return {PipelineStageName::Color_Adjustment, OperatorType::SATURATION};
    case AdjustmentField::RawDecode:
      return {PipelineStageName::Image_Loading, OperatorType::RAW_DECODE};
    case AdjustmentField::LensCalib:
      return {PipelineStageName::Image_Loading, OperatorType::LENS_CALIBRATION};
    case AdjustmentField::ColorTemp:
      return {PipelineStageName::To_WorkingSpace, OperatorType::COLOR_TEMP};
    case AdjustmentField::Hls:
      return {PipelineStageName::Color_Adjustment, OperatorType::HLS};
    case AdjustmentField::ColorWheel:
      return {PipelineStageName::Color_Adjustment, OperatorType::COLOR_WHEEL};
    case AdjustmentField::Blacks:
      return {PipelineStageName::Basic_Adjustment, OperatorType::BLACK};
    case AdjustmentField::Whites:
      return {PipelineStageName::Basic_Adjustment, OperatorType::WHITE};
    case AdjustmentField::Shadows:
      return {PipelineStageName::Basic_Adjustment, OperatorType::SHADOWS};
    case AdjustmentField::Highlights:
      return {PipelineStageName::Basic_Adjustment, OperatorType::HIGHLIGHTS};
    case AdjustmentField::Curve:
      return {PipelineStageName::Basic_Adjustment, OperatorType::CURVE};
    case AdjustmentField::Sharpen:
      return {PipelineStageName::Detail_Adjustment, OperatorType::SHARPEN};
    case AdjustmentField::Clarity:
      return {PipelineStageName::Detail_Adjustment, OperatorType::CLARITY};
    case AdjustmentField::Lut:
      return {PipelineStageName::Color_Adjustment, OperatorType::LMT};
    case AdjustmentField::CropRotate:
      return {PipelineStageName::Geometry_Adjustment, OperatorType::CROP_ROTATE};
  }
  return {PipelineStageName::Basic_Adjustment, OperatorType::EXPOSURE};
}

// =========================================================================
// ParamsForField
// =========================================================================

auto ParamsForField(AdjustmentField field, const AdjustmentState& s,
                    CPUPipelineExecutor* exec) -> nlohmann::json {
  using namespace hls;
  using namespace color_temp;
  using curve::CurveControlPointsToParams;
  using geometry::ClampCropRect;

  switch (field) {
    case AdjustmentField::Exposure:
      return {{"exposure", s.exposure_}};
    case AdjustmentField::Contrast:
      return {{"contrast", s.contrast_}};
    case AdjustmentField::Saturation:
      return {{"saturation", s.saturation_}};
    case AdjustmentField::RawDecode: {
      const nlohmann::json defaults = pipeline_defaults::MakeDefaultRawDecodeParams();
      nlohmann::json params =
          exec ? ReadCurrentOperatorParams(*exec, PipelineStageName::Image_Loading,
                                           OperatorType::RAW_DECODE)
                     .value_or(defaults)
               : defaults;
      if (!params.is_object()) {
        params = defaults;
      }
      if (!params.contains("raw") || !params["raw"].is_object()) {
        params["raw"] = defaults.value("raw", nlohmann::json::object());
      }
      if (defaults.contains("raw") && defaults["raw"].is_object()) {
        for (auto it = defaults["raw"].begin(); it != defaults["raw"].end(); ++it) {
          if (!params["raw"].contains(it.key())) {
            params["raw"][it.key()] = it.value();
          }
        }
      }
      params["raw"]["highlights_reconstruct"] = s.raw_highlights_reconstruct_;
      return params;
    }
    case AdjustmentField::LensCalib: {
      const nlohmann::json defaults = pipeline_defaults::MakeDefaultLensCalibParams();
      nlohmann::json params =
          exec ? ReadCurrentOperatorParams(*exec, PipelineStageName::Image_Loading,
                                           OperatorType::LENS_CALIBRATION)
                     .value_or(defaults)
               : defaults;
      if (!params.is_object()) {
        params = defaults;
      }
      if (!params.contains("lens_calib") || !params["lens_calib"].is_object()) {
        params["lens_calib"] = defaults.value("lens_calib", nlohmann::json::object());
      }
      if (defaults.contains("lens_calib") && defaults["lens_calib"].is_object()) {
        for (auto it = defaults["lens_calib"].begin(); it != defaults["lens_calib"].end(); ++it) {
          if (!params["lens_calib"].contains(it.key())) {
            params["lens_calib"][it.key()] = it.value();
          }
        }
      }
      params["lens_calib"]["enabled"]    = s.lens_calib_enabled_;
      params["lens_calib"]["lens_maker"] = s.lens_override_make_;
      params["lens_calib"]["lens_model"] =
          s.lens_override_make_.empty() ? std::string{} : s.lens_override_model_;
      return params;
    }
    case AdjustmentField::ColorTemp:
      return {{"color_temp",
               {{"mode", ColorTempModeToString(s.color_temp_mode_)},
                {"cct", std::clamp(s.color_temp_custom_cct_,
                                   static_cast<float>(kCctMin),
                                   static_cast<float>(kCctMax))},
                {"tint", std::clamp(s.color_temp_custom_tint_,
                                    static_cast<float>(kTintMin),
                                    static_cast<float>(kTintMax))},
                {"resolved_cct", std::clamp(s.color_temp_resolved_cct_,
                                            static_cast<float>(kCctMin),
                                            static_cast<float>(kCctMax))},
                {"resolved_tint", std::clamp(s.color_temp_resolved_tint_,
                                             static_cast<float>(kTintMin),
                                             static_cast<float>(kTintMax))}}}};
    case AdjustmentField::Hls: {
      nlohmann::json hue_bins      = nlohmann::json::array();
      nlohmann::json hls_adj_table = nlohmann::json::array();
      nlohmann::json h_range_table = nlohmann::json::array();
      for (size_t i = 0; i < kCandidateHues.size(); ++i) {
        hue_bins.push_back(kCandidateHues[i]);
        hls_adj_table.push_back(std::array<float, 3>{
            std::clamp(s.hls_hue_adjust_table_[i], -kMaxHueShiftDegrees, kMaxHueShiftDegrees),
            std::clamp(s.hls_lightness_adjust_table_[i], kAdjUiMin, kAdjUiMax) /
                kAdjUiToParamScale,
            std::clamp(s.hls_saturation_adjust_table_[i], kAdjUiMin, kAdjUiMax) /
                kAdjUiToParamScale});
        h_range_table.push_back(std::max(s.hls_hue_range_table_[i], 1.0f));
      }
      const int active_idx = ActiveHlsProfileIndex(s);

      return {{"HLS",
               {{"hue_bins", std::move(hue_bins)},
                {"hls_adj_table", std::move(hls_adj_table)},
                {"h_range_table", std::move(h_range_table)},
                {"target_hls",
                 std::array<float, 3>{WrapHueDegrees(s.hls_target_hue_), kFixedTargetLightness,
                                      kFixedTargetSaturation}},
                {"hls_adj",
                 std::array<float, 3>{
                     std::clamp(s.hls_hue_adjust_table_[active_idx], -kMaxHueShiftDegrees,
                                kMaxHueShiftDegrees),
                     std::clamp(s.hls_lightness_adjust_table_[active_idx], kAdjUiMin, kAdjUiMax) /
                         kAdjUiToParamScale,
                     std::clamp(s.hls_saturation_adjust_table_[active_idx], kAdjUiMin, kAdjUiMax) /
                         kAdjUiToParamScale}},
                {"h_range", std::max(s.hls_hue_range_table_[active_idx], 1.0f)},
                {"l_range", kFixedLightnessRange},
                {"s_range", kFixedSaturationRange}}}};
    }
    case AdjustmentField::ColorWheel: {
      CdlWheelState lift  = s.lift_wheel_;
      CdlWheelState gamma = s.gamma_wheel_;
      CdlWheelState gain  = s.gain_wheel_;
      UpdateCdlWheelDerivedColor(lift, false, false);
      UpdateCdlWheelDerivedColor(gamma, true, true);
      UpdateCdlWheelDerivedColor(gain, true, false);

      auto WheelToJson = [](const CdlWheelState& wheel) -> nlohmann::json {
        return {
            {"disc",
             {{"x", static_cast<float>(wheel.disc_position_.x())},
              {"y", static_cast<float>(wheel.disc_position_.y())}}},
            {"strength", wheel.strength_},
            {"color_offset",
             {{"x", wheel.color_offset_[0]},
              {"y", wheel.color_offset_[1]},
              {"z", wheel.color_offset_[2]}}},
            {"luminance_offset", std::clamp(wheel.master_offset_, -1.0f, 1.0f)},
        };
      };

      return {{"color_wheel",
               {{"lift", WheelToJson(lift)},
                {"gamma", WheelToJson(gamma)},
                {"gain", WheelToJson(gain)}}}};
    }
    case AdjustmentField::Blacks:
      return {{"black", s.blacks_}};
    case AdjustmentField::Whites:
      return {{"white", s.whites_}};
    case AdjustmentField::Shadows:
      return {{"shadows", s.shadows_}};
    case AdjustmentField::Highlights:
      return {{"highlights", s.highlights_}};
    case AdjustmentField::Curve:
      return CurveControlPointsToParams(s.curve_points_);
    case AdjustmentField::Sharpen:
      return {{"sharpen", {{"offset", s.sharpen_}}}};
    case AdjustmentField::Clarity:
      return {{"clarity", s.clarity_}};
    case AdjustmentField::Lut:
      return {{"ocio_lmt", s.lut_path_}};
    case AdjustmentField::CropRotate: {
      const auto crop_rect = ClampCropRect(s.crop_x_, s.crop_y_, s.crop_w_, s.crop_h_);
      const bool has_rotation = std::abs(s.rotate_degrees_) > 1e-4f;
      const bool has_crop =
          s.crop_enabled_ &&
          (std::abs(crop_rect[0]) > 1e-4f || std::abs(crop_rect[1]) > 1e-4f ||
           std::abs(crop_rect[2] - 1.0f) > 1e-4f || std::abs(crop_rect[3] - 1.0f) > 1e-4f);
      return {{"crop_rotate",
               {{"enabled", has_rotation || has_crop},
                {"angle_degrees", s.rotate_degrees_},
                {"enable_crop", s.crop_enabled_},
                {"crop_rect",
                 {{"x", crop_rect[0]},
                  {"y", crop_rect[1]},
                  {"w", crop_rect[2]},
                  {"h", crop_rect[3]}}},
                {"expand_to_fit", s.crop_expand_to_fit_}}}};
    }
  }
  return {};
}

// =========================================================================
// FieldChanged
// =========================================================================

auto FieldChanged(AdjustmentField field, const AdjustmentState& current,
                  const AdjustmentState& committed) -> bool {
  using color_wheel::ClampDiscPoint;
  using curve::CurveControlPointsEqual;
  using geometry::ClampCropRect;

  switch (field) {
    case AdjustmentField::Exposure:
      return !NearlyEqual(current.exposure_, committed.exposure_);
    case AdjustmentField::Contrast:
      return !NearlyEqual(current.contrast_, committed.contrast_);
    case AdjustmentField::Saturation:
      return !NearlyEqual(current.saturation_, committed.saturation_);
    case AdjustmentField::RawDecode:
      return current.raw_highlights_reconstruct_ != committed.raw_highlights_reconstruct_;
    case AdjustmentField::LensCalib:
      return current.lens_calib_enabled_ != committed.lens_calib_enabled_ ||
             current.lens_override_make_ != committed.lens_override_make_ ||
             current.lens_override_model_ != committed.lens_override_model_;
    case AdjustmentField::ColorTemp:
      return current.color_temp_mode_ != committed.color_temp_mode_ ||
             !NearlyEqual(current.color_temp_custom_cct_, committed.color_temp_custom_cct_) ||
             !NearlyEqual(current.color_temp_custom_tint_, committed.color_temp_custom_tint_);
    case AdjustmentField::Hls:
      for (size_t i = 0; i < hls::kCandidateHues.size(); ++i) {
        if (!NearlyEqual(current.hls_hue_adjust_table_[i],
                         committed.hls_hue_adjust_table_[i]) ||
            !NearlyEqual(current.hls_lightness_adjust_table_[i],
                         committed.hls_lightness_adjust_table_[i]) ||
            !NearlyEqual(current.hls_saturation_adjust_table_[i],
                         committed.hls_saturation_adjust_table_[i]) ||
            !NearlyEqual(current.hls_hue_range_table_[i],
                         committed.hls_hue_range_table_[i])) {
          return true;
        }
      }
      return false;
    case AdjustmentField::ColorWheel: {
      auto WheelChanged = [](const CdlWheelState& a, const CdlWheelState& b) -> bool {
        return !NearlyEqual(static_cast<float>(a.disc_position_.x()),
                            static_cast<float>(b.disc_position_.x())) ||
               !NearlyEqual(static_cast<float>(a.disc_position_.y()),
                            static_cast<float>(b.disc_position_.y())) ||
               !NearlyEqual(a.master_offset_, b.master_offset_) ||
               !NearlyEqual(a.strength_, b.strength_) ||
               !NearlyEqual(a.color_offset_[0], b.color_offset_[0]) ||
               !NearlyEqual(a.color_offset_[1], b.color_offset_[1]) ||
               !NearlyEqual(a.color_offset_[2], b.color_offset_[2]);
      };
      return WheelChanged(current.lift_wheel_, committed.lift_wheel_) ||
             WheelChanged(current.gamma_wheel_, committed.gamma_wheel_) ||
             WheelChanged(current.gain_wheel_, committed.gain_wheel_);
    }
    case AdjustmentField::Blacks:
      return !NearlyEqual(current.blacks_, committed.blacks_);
    case AdjustmentField::Whites:
      return !NearlyEqual(current.whites_, committed.whites_);
    case AdjustmentField::Shadows:
      return !NearlyEqual(current.shadows_, committed.shadows_);
    case AdjustmentField::Highlights:
      return !NearlyEqual(current.highlights_, committed.highlights_);
    case AdjustmentField::Curve:
      return !CurveControlPointsEqual(current.curve_points_, committed.curve_points_);
    case AdjustmentField::Sharpen:
      return !NearlyEqual(current.sharpen_, committed.sharpen_);
    case AdjustmentField::Clarity:
      return !NearlyEqual(current.clarity_, committed.clarity_);
    case AdjustmentField::Lut:
      return current.lut_path_ != committed.lut_path_;
    case AdjustmentField::CropRotate: {
      const auto state_rect =
          ClampCropRect(current.crop_x_, current.crop_y_, current.crop_w_, current.crop_h_);
      const auto committed_rect = ClampCropRect(committed.crop_x_, committed.crop_y_,
                                                committed.crop_w_, committed.crop_h_);
      return !NearlyEqual(current.rotate_degrees_, committed.rotate_degrees_) ||
             current.crop_enabled_ != committed.crop_enabled_ ||
             current.crop_expand_to_fit_ != committed.crop_expand_to_fit_ ||
             !NearlyEqual(state_rect[0], committed_rect[0]) ||
             !NearlyEqual(state_rect[1], committed_rect[1]) ||
             !NearlyEqual(state_rect[2], committed_rect[2]) ||
             !NearlyEqual(state_rect[3], committed_rect[3]);
    }
  }
  return false;
}

// =========================================================================
// LoadStateFromPipeline
// =========================================================================

auto LoadStateFromPipeline(CPUPipelineExecutor& exec,
                           const AdjustmentState& base_state)
    -> std::pair<AdjustmentState, bool> {
  using color_wheel::ClampDiscPoint;
  using geometry::ClampCropRect;
  using hls::ClosestCandidateHueIndex;
  using hls::MakeFilledArray;
  using hls::WrapHueDegrees;

  AdjustmentState loaded_state     = base_state;
  loaded_state.type_               = base_state.type_;
  loaded_state.rotate_degrees_     = 0.0f;
  loaded_state.crop_enabled_       = true;
  loaded_state.crop_x_             = 0.0f;
  loaded_state.crop_y_             = 0.0f;
  loaded_state.crop_w_             = 1.0f;
  loaded_state.crop_h_             = 1.0f;
  loaded_state.crop_expand_to_fit_ = true;
  bool has_loaded_any              = false;

  const auto& loading  = exec.GetStage(PipelineStageName::Image_Loading);
  const auto& geometry = exec.GetStage(PipelineStageName::Geometry_Adjustment);
  const auto& to_ws    = exec.GetStage(PipelineStageName::To_WorkingSpace);
  const auto& basic    = exec.GetStage(PipelineStageName::Basic_Adjustment);
  const auto& color    = exec.GetStage(PipelineStageName::Color_Adjustment);
  const auto& detail   = exec.GetStage(PipelineStageName::Detail_Adjustment);

  // --- Raw decode ---
  if (const auto raw_json = ReadNestedObject(loading, OperatorType::RAW_DECODE, "raw");
      raw_json.has_value()) {
    const auto& raw = *raw_json;
    if (raw.contains("highlights_reconstruct")) {
      try {
        loaded_state.raw_highlights_reconstruct_ = raw["highlights_reconstruct"].get<bool>();
      } catch (...) {
      }
    }
    has_loaded_any = true;
  }

  // --- Lens calibration ---
  if (const auto lens_json =
          ReadNestedObject(loading, OperatorType::LENS_CALIBRATION, "lens_calib");
      lens_json.has_value()) {
    const auto& lens = *lens_json;
    if (lens.contains("enabled")) {
      try {
        loaded_state.lens_calib_enabled_ = lens["enabled"].get<bool>();
      } catch (...) {
      }
    }
    if (lens.contains("lens_maker")) {
      try {
        loaded_state.lens_override_make_ = lens["lens_maker"].get<std::string>();
      } catch (...) {
      }
    }
    if (lens.contains("lens_model")) {
      try {
        loaded_state.lens_override_model_ = lens["lens_model"].get<std::string>();
      } catch (...) {
      }
    }
    if (loaded_state.lens_override_make_.empty()) {
      loaded_state.lens_override_model_.clear();
    }
    has_loaded_any = true;
  }

  // --- Basic adjustments ---
  if (const auto v = ReadFloat(basic, OperatorType::EXPOSURE, "exposure"); v.has_value()) {
    loaded_state.exposure_ = v.value();
    has_loaded_any         = true;
  }
  if (const auto v = ReadFloat(basic, OperatorType::CONTRAST, "contrast"); v.has_value()) {
    loaded_state.contrast_ = v.value();
    has_loaded_any         = true;
  }

  const auto black_enabled = IsOperatorEnabled(basic, OperatorType::BLACK);
  if (black_enabled.has_value() && black_enabled.value()) {
    loaded_state.blacks_ = exec.GetGlobalParams().black_point_ * kBlackSliderFromGlobalScale;
    has_loaded_any       = true;
  } else if (const auto v = ReadFloat(basic, OperatorType::BLACK, "black"); v.has_value()) {
    loaded_state.blacks_ = v.value();
    has_loaded_any       = true;
  }

  const auto white_enabled = IsOperatorEnabled(basic, OperatorType::WHITE);
  if (white_enabled.has_value() && white_enabled.value()) {
    loaded_state.whites_ =
        (exec.GetGlobalParams().white_point_ - 1.0f) * kWhiteSliderFromGlobalScale;
    has_loaded_any = true;
  } else if (const auto v = ReadFloat(basic, OperatorType::WHITE, "white"); v.has_value()) {
    loaded_state.whites_ = v.value();
    has_loaded_any       = true;
  }

  const auto shadows_enabled = IsOperatorEnabled(basic, OperatorType::SHADOWS);
  if (shadows_enabled.has_value() && shadows_enabled.value()) {
    loaded_state.shadows_ =
        exec.GetGlobalParams().shadows_offset_ * kShadowsSliderFromGlobalScale;
    has_loaded_any = true;
  } else if (const auto v = ReadFloat(basic, OperatorType::SHADOWS, "shadows"); v.has_value()) {
    loaded_state.shadows_ = v.value();
    has_loaded_any        = true;
  }

  const auto highlights_enabled = IsOperatorEnabled(basic, OperatorType::HIGHLIGHTS);
  if (highlights_enabled.has_value() && highlights_enabled.value()) {
    loaded_state.highlights_ =
        exec.GetGlobalParams().highlights_offset_ * kHighlightsSliderFromGlobalScale;
    has_loaded_any = true;
  } else if (const auto v = ReadFloat(basic, OperatorType::HIGHLIGHTS, "highlights");
             v.has_value()) {
    loaded_state.highlights_ = v.value();
    has_loaded_any           = true;
  }

  // --- Curve ---
  if (const auto curve_points = ReadCurvePoints(basic, OperatorType::CURVE);
      curve_points.has_value()) {
    loaded_state.curve_points_ = curve::NormalizeCurveControlPoints(*curve_points);
    has_loaded_any             = true;
  }

  // --- Saturation ---
  if (const auto v = ReadFloat(color, OperatorType::SATURATION, "saturation"); v.has_value()) {
    loaded_state.saturation_ = v.value();
    has_loaded_any           = true;
  }

  // --- Color wheel (CDL) ---
  if (const auto color_wheel_json =
          ReadNestedObject(color, OperatorType::COLOR_WHEEL, "color_wheel");
      color_wheel_json.has_value()) {
    auto ParseWheel = [](const nlohmann::json& wheels, const char* key, CdlWheelState& wheel,
                         bool add_unity, bool invert_delta) -> bool {
      if (!wheels.contains(key) || !wheels.at(key).is_object()) {
        return false;
      }
      const auto& src            = wheels.at(key);
      bool        loaded_any_field = false;
      bool        has_color_offset = false;

      if (src.contains("disc") && src["disc"].is_object() && src["disc"].contains("x") &&
          src["disc"].contains("y")) {
        try {
          const QPointF disc(src["disc"]["x"].get<float>(), src["disc"]["y"].get<float>());
          wheel.disc_position_ = ClampDiscPoint(disc);
          loaded_any_field     = true;
        } catch (...) {
        }
      }

      if (src.contains("strength")) {
        try {
          wheel.strength_ =
              std::clamp(src["strength"].get<float>(), 0.0f, color_wheel::kStrengthDefault);
          loaded_any_field = true;
        } catch (...) {
        }
      }

      if (src.contains("luminance_offset")) {
        try {
          wheel.master_offset_ = std::clamp(src["luminance_offset"].get<float>(), -1.0f, 1.0f);
          loaded_any_field     = true;
        } catch (...) {
        }
      }

      if (src.contains("color_offset") && src["color_offset"].is_object() &&
          src["color_offset"].contains("x") && src["color_offset"].contains("y") &&
          src["color_offset"].contains("z")) {
        try {
          wheel.color_offset_[0] = src["color_offset"]["x"].get<float>();
          wheel.color_offset_[1] = src["color_offset"]["y"].get<float>();
          wheel.color_offset_[2] = src["color_offset"]["z"].get<float>();
          has_color_offset       = true;
          loaded_any_field       = true;
        } catch (...) {
        }
      }

      if (!has_color_offset) {
        UpdateCdlWheelDerivedColor(wheel, add_unity, invert_delta);
      }
      return loaded_any_field;
    };

    bool loaded_color_wheel = false;
    loaded_color_wheel |=
        ParseWheel(*color_wheel_json, "lift", loaded_state.lift_wheel_, false, false);
    loaded_color_wheel |=
        ParseWheel(*color_wheel_json, "gamma", loaded_state.gamma_wheel_, true, true);
    loaded_color_wheel |=
        ParseWheel(*color_wheel_json, "gain", loaded_state.gain_wheel_, true, false);
    if (loaded_color_wheel) {
      has_loaded_any = true;
    }
  }

  // --- Color temperature ---
  if (const auto color_temp_json =
          ReadNestedObject(to_ws, OperatorType::COLOR_TEMP, "color_temp");
      color_temp_json.has_value()) {
    const auto& ct = *color_temp_json;
    if (ct.contains("mode") && ct["mode"].is_string()) {
      loaded_state.color_temp_mode_ = ParseColorTempMode(ct["mode"].get<std::string>());
    }
    if (ct.contains("cct")) {
      try {
        loaded_state.color_temp_custom_cct_ = std::clamp(
            ct["cct"].get<float>(), static_cast<float>(color_temp::kCctMin),
            static_cast<float>(color_temp::kCctMax));
      } catch (...) {
      }
    }
    if (ct.contains("tint")) {
      try {
        loaded_state.color_temp_custom_tint_ = std::clamp(
            ct["tint"].get<float>(), static_cast<float>(color_temp::kTintMin),
            static_cast<float>(color_temp::kTintMax));
      } catch (...) {
      }
    }
    if (ct.contains("resolved_cct")) {
      try {
        loaded_state.color_temp_resolved_cct_ = std::clamp(
            ct["resolved_cct"].get<float>(), static_cast<float>(color_temp::kCctMin),
            static_cast<float>(color_temp::kCctMax));
      } catch (...) {
      }
    }
    if (ct.contains("resolved_tint")) {
      try {
        loaded_state.color_temp_resolved_tint_ = std::clamp(
            ct["resolved_tint"].get<float>(), static_cast<float>(color_temp::kTintMin),
            static_cast<float>(color_temp::kTintMax));
      } catch (...) {
      }
    }
    has_loaded_any = true;
  }

  // --- HLS ---
  if (const auto hls_json = ReadNestedObject(color, OperatorType::HLS, "HLS");
      hls_json.has_value()) {
    auto ReadArray3 = [](const nlohmann::json& obj, const char* key,
                         std::array<float, 3>& out) -> bool {
      if (!obj.contains(key) || !obj[key].is_array() || obj[key].size() < 3) {
        return false;
      }
      try {
        out[0] = obj[key][0].get<float>();
        out[1] = obj[key][1].get<float>();
        out[2] = obj[key][2].get<float>();
        return true;
      } catch (...) {
        return false;
      }
    };

    const auto& hls_data = *hls_json;
    loaded_state.hls_hue_adjust_table_.fill(0.0f);
    loaded_state.hls_lightness_adjust_table_.fill(0.0f);
    loaded_state.hls_saturation_adjust_table_.fill(0.0f);
    loaded_state.hls_hue_range_table_ = MakeFilledArray(hls::kDefaultHueRange);
    std::array<float, 3> target_hls   = {loaded_state.hls_target_hue_, hls::kFixedTargetLightness,
                                          hls::kFixedTargetSaturation};
    std::array<float, 3> hls_adj      = {};
    bool                 has_adj_table   = false;
    bool                 has_range_table = false;

    if (hls_data.contains("hls_adj_table") && hls_data["hls_adj_table"].is_array()) {
      const auto& adj_tbl  = hls_data["hls_adj_table"];
      const bool  has_bins = hls_data.contains("hue_bins") && hls_data["hue_bins"].is_array();
      for (int i = 0; i < static_cast<int>(adj_tbl.size()); ++i) {
        if (!adj_tbl[i].is_array() || adj_tbl[i].size() < 3) {
          continue;
        }
        int idx = i;
        if (has_bins && i < static_cast<int>(hls_data["hue_bins"].size())) {
          try {
            idx = ClosestCandidateHueIndex(hls_data["hue_bins"][i].get<float>());
          } catch (...) {
          }
        }
        if (idx < 0 || idx >= static_cast<int>(hls::kCandidateHues.size())) {
          continue;
        }
        try {
          loaded_state.hls_hue_adjust_table_[idx] = std::clamp(
              adj_tbl[i][0].get<float>(), -hls::kMaxHueShiftDegrees, hls::kMaxHueShiftDegrees);
          loaded_state.hls_lightness_adjust_table_[idx] = std::clamp(
              adj_tbl[i][1].get<float>() * hls::kAdjUiToParamScale, hls::kAdjUiMin,
              hls::kAdjUiMax);
          loaded_state.hls_saturation_adjust_table_[idx] = std::clamp(
              adj_tbl[i][2].get<float>() * hls::kAdjUiToParamScale, hls::kAdjUiMin,
              hls::kAdjUiMax);
          has_adj_table = true;
        } catch (...) {
        }
      }
    }

    if (hls_data.contains("h_range_table") && hls_data["h_range_table"].is_array()) {
      const auto& range_tbl = hls_data["h_range_table"];
      const bool  has_bins  = hls_data.contains("hue_bins") && hls_data["hue_bins"].is_array();
      for (int i = 0; i < static_cast<int>(range_tbl.size()); ++i) {
        int idx = i;
        if (has_bins && i < static_cast<int>(hls_data["hue_bins"].size())) {
          try {
            idx = ClosestCandidateHueIndex(hls_data["hue_bins"][i].get<float>());
          } catch (...) {
          }
        }
        if (idx < 0 || idx >= static_cast<int>(hls::kCandidateHues.size())) {
          continue;
        }
        try {
          loaded_state.hls_hue_range_table_[idx] =
              std::clamp(range_tbl[i].get<float>(), 1.0f, 180.0f);
          has_range_table = true;
        } catch (...) {
        }
      }
    }

    (void)ReadArray3(hls_data, "target_hls", target_hls);
    (void)ReadArray3(hls_data, "hls_adj", hls_adj);

    loaded_state.hls_target_hue_ = WrapHueDegrees(target_hls[0]);
    const int active_idx         = ActiveHlsProfileIndex(loaded_state);
    loaded_state.hls_target_hue_ = hls::kCandidateHues[static_cast<size_t>(active_idx)];
    if (!has_adj_table) {
      loaded_state.hls_hue_adjust_table_[active_idx] =
          std::clamp(hls_adj[0], -hls::kMaxHueShiftDegrees, hls::kMaxHueShiftDegrees);
      loaded_state.hls_lightness_adjust_table_[active_idx] =
          std::clamp(hls_adj[1] * hls::kAdjUiToParamScale, hls::kAdjUiMin, hls::kAdjUiMax);
      loaded_state.hls_saturation_adjust_table_[active_idx] =
          std::clamp(hls_adj[2] * hls::kAdjUiToParamScale, hls::kAdjUiMin, hls::kAdjUiMax);
    }

    if (!has_range_table && hls_data.contains("h_range")) {
      try {
        loaded_state.hls_hue_range_table_[active_idx] =
            std::clamp(hls_data["h_range"].get<float>(), 1.0f, 180.0f);
      } catch (...) {
      }
    }
    LoadActiveHlsProfile(loaded_state);
    has_loaded_any = true;
  }

  // --- Detail ---
  if (const auto v = ReadNestedFloat(detail, OperatorType::SHARPEN, "sharpen", "offset");
      v.has_value()) {
    loaded_state.sharpen_ = v.value();
    has_loaded_any        = true;
  }
  if (const auto v = ReadFloat(detail, OperatorType::CLARITY, "clarity"); v.has_value()) {
    loaded_state.clarity_ = v.value();
    has_loaded_any        = true;
  }

  // --- Crop / Rotate ---
  if (const auto crop_rotate_json =
          ReadNestedObject(geometry, OperatorType::CROP_ROTATE, "crop_rotate");
      crop_rotate_json.has_value()) {
    const auto& crop_rotate    = *crop_rotate_json;
    loaded_state.rotate_degrees_ = crop_rotate.value("angle_degrees", loaded_state.rotate_degrees_);
    loaded_state.crop_enabled_ = crop_rotate.value("enable_crop", loaded_state.crop_enabled_);
    loaded_state.crop_expand_to_fit_ =
        crop_rotate.value("expand_to_fit", loaded_state.crop_expand_to_fit_);
    bool has_non_full_crop_rect = false;
    if (crop_rotate.contains("crop_rect") && crop_rotate["crop_rect"].is_object()) {
      const auto& crop_rect = crop_rotate["crop_rect"];
      const auto  clamped   = ClampCropRect(crop_rect.value("x", loaded_state.crop_x_),
                                             crop_rect.value("y", loaded_state.crop_y_),
                                             crop_rect.value("w", loaded_state.crop_w_),
                                             crop_rect.value("h", loaded_state.crop_h_));
      loaded_state.crop_x_ = clamped[0];
      loaded_state.crop_y_ = clamped[1];
      loaded_state.crop_w_ = clamped[2];
      loaded_state.crop_h_ = clamped[3];
      has_non_full_crop_rect = std::abs(loaded_state.crop_x_) > 1e-4f ||
                               std::abs(loaded_state.crop_y_) > 1e-4f ||
                               std::abs(loaded_state.crop_w_ - 1.0f) > 1e-4f ||
                               std::abs(loaded_state.crop_h_ - 1.0f) > 1e-4f;
    }
    loaded_state.crop_enabled_ = loaded_state.crop_enabled_ || has_non_full_crop_rect;
    has_loaded_any             = true;
  }

  // --- LUT ---
  const auto lut = ReadString(color, OperatorType::LMT, "ocio_lmt");
  if (lut.has_value()) {
    loaded_state.lut_path_ = *lut;
    has_loaded_any         = true;
  } else if (const auto lmt_enabled = IsOperatorEnabled(color, OperatorType::LMT);
             lmt_enabled.has_value() && !lmt_enabled.value()) {
    loaded_state.lut_path_.clear();
    has_loaded_any = true;
  }

  // --- Resolved color temperature from global params ---
  loaded_state.color_temp_resolved_cct_ = std::clamp(
      exec.GetGlobalParams().color_temp_resolved_cct_,
      static_cast<float>(color_temp::kCctMin), static_cast<float>(color_temp::kCctMax));
  loaded_state.color_temp_resolved_tint_ = std::clamp(
      exec.GetGlobalParams().color_temp_resolved_tint_,
      static_cast<float>(color_temp::kTintMin), static_cast<float>(color_temp::kTintMax));
  loaded_state.color_temp_supported_ = exec.GetGlobalParams().color_temp_matrices_valid_;

  return {loaded_state, has_loaded_any};
}

}  // namespace puerhlab::ui::pipeline_io
