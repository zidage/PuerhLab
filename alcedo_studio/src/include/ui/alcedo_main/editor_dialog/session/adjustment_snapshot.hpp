//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/color_temp_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/display_transform_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/geometry_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/look_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/raw_decode_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/tone_adjustment_state.hpp"

namespace alcedo::ui {

struct EditorAdjustmentSnapshot {
  ToneAdjustmentState tone_;
  ColorTempAdjustmentState color_temp_;
  LookAdjustmentState look_;
  DisplayTransformAdjustmentState display_transform_;
  GeometryAdjustmentState geometry_;
  RawDecodeAdjustmentState raw_;
  RenderType type_ = RenderType::FAST_PREVIEW;
};

inline auto ToLegacyAdjustmentState(const EditorAdjustmentSnapshot& snapshot) -> AdjustmentState {
  AdjustmentState legacy;
  // Tone
  legacy.exposure_ = snapshot.tone_.exposure_;
  legacy.contrast_ = snapshot.tone_.contrast_;
  legacy.blacks_ = snapshot.tone_.blacks_;
  legacy.whites_ = snapshot.tone_.whites_;
  legacy.shadows_ = snapshot.tone_.shadows_;
  legacy.highlights_ = snapshot.tone_.highlights_;
  legacy.curve_points_ = snapshot.tone_.curve_points_;
  legacy.saturation_ = snapshot.tone_.saturation_;
  legacy.sharpen_ = snapshot.tone_.sharpen_;
  legacy.clarity_ = snapshot.tone_.clarity_;
  // Color temp
  legacy.color_temp_mode_ = snapshot.color_temp_.mode_;
  legacy.color_temp_custom_cct_ = snapshot.color_temp_.custom_cct_;
  legacy.color_temp_custom_tint_ = snapshot.color_temp_.custom_tint_;
  legacy.color_temp_resolved_cct_ = snapshot.color_temp_.resolved_cct_;
  legacy.color_temp_resolved_tint_ = snapshot.color_temp_.resolved_tint_;
  legacy.color_temp_supported_ = snapshot.color_temp_.supported_;
  // Look
  legacy.hls_target_hue_ = snapshot.look_.hls_target_hue_;
  legacy.hls_hue_adjust_ = snapshot.look_.hls_hue_adjust_;
  legacy.hls_lightness_adjust_ = snapshot.look_.hls_lightness_adjust_;
  legacy.hls_saturation_adjust_ = snapshot.look_.hls_saturation_adjust_;
  legacy.hls_hue_range_ = snapshot.look_.hls_hue_range_;
  legacy.lift_wheel_ = snapshot.look_.lift_wheel_;
  legacy.gamma_wheel_ = snapshot.look_.gamma_wheel_;
  legacy.gain_wheel_ = snapshot.look_.gain_wheel_;
  legacy.hls_hue_adjust_table_ = snapshot.look_.hls_hue_adjust_table_;
  legacy.hls_lightness_adjust_table_ = snapshot.look_.hls_lightness_adjust_table_;
  legacy.hls_saturation_adjust_table_ = snapshot.look_.hls_saturation_adjust_table_;
  legacy.hls_hue_range_table_ = snapshot.look_.hls_hue_range_table_;
  legacy.lut_path_ = snapshot.look_.lut_path_;
  // Display transform
  legacy.odt_ = snapshot.display_transform_.odt_;
  // Geometry
  legacy.rotate_degrees_ = snapshot.geometry_.rotate_degrees_;
  legacy.crop_enabled_ = snapshot.geometry_.crop_enabled_;
  legacy.crop_x_ = snapshot.geometry_.crop_x_;
  legacy.crop_y_ = snapshot.geometry_.crop_y_;
  legacy.crop_w_ = snapshot.geometry_.crop_w_;
  legacy.crop_h_ = snapshot.geometry_.crop_h_;
  legacy.crop_expand_to_fit_ = snapshot.geometry_.crop_expand_to_fit_;
  legacy.crop_aspect_preset_ = snapshot.geometry_.crop_aspect_preset_;
  legacy.crop_aspect_width_ = snapshot.geometry_.crop_aspect_width_;
  legacy.crop_aspect_height_ = snapshot.geometry_.crop_aspect_height_;
  // Raw decode
  legacy.raw_highlights_reconstruct_ = snapshot.raw_.raw_highlights_reconstruct_;
  legacy.lens_calib_enabled_ = snapshot.raw_.lens_calib_enabled_;
  legacy.lens_override_make_ = snapshot.raw_.lens_override_make_;
  legacy.lens_override_model_ = snapshot.raw_.lens_override_model_;
  // Render type
  legacy.type_ = snapshot.type_;
  return legacy;
}

inline auto FromLegacyAdjustmentState(const AdjustmentState& state) -> EditorAdjustmentSnapshot {
  EditorAdjustmentSnapshot snapshot;
  // Tone
  snapshot.tone_.exposure_ = state.exposure_;
  snapshot.tone_.contrast_ = state.contrast_;
  snapshot.tone_.blacks_ = state.blacks_;
  snapshot.tone_.whites_ = state.whites_;
  snapshot.tone_.shadows_ = state.shadows_;
  snapshot.tone_.highlights_ = state.highlights_;
  snapshot.tone_.curve_points_ = state.curve_points_;
  snapshot.tone_.saturation_ = state.saturation_;
  snapshot.tone_.sharpen_ = state.sharpen_;
  snapshot.tone_.clarity_ = state.clarity_;
  // Color temp
  snapshot.color_temp_.mode_ = state.color_temp_mode_;
  snapshot.color_temp_.custom_cct_ = state.color_temp_custom_cct_;
  snapshot.color_temp_.custom_tint_ = state.color_temp_custom_tint_;
  snapshot.color_temp_.resolved_cct_ = state.color_temp_resolved_cct_;
  snapshot.color_temp_.resolved_tint_ = state.color_temp_resolved_tint_;
  snapshot.color_temp_.supported_ = state.color_temp_supported_;
  // Look
  snapshot.look_.hls_target_hue_ = state.hls_target_hue_;
  snapshot.look_.hls_hue_adjust_ = state.hls_hue_adjust_;
  snapshot.look_.hls_lightness_adjust_ = state.hls_lightness_adjust_;
  snapshot.look_.hls_saturation_adjust_ = state.hls_saturation_adjust_;
  snapshot.look_.hls_hue_range_ = state.hls_hue_range_;
  snapshot.look_.lift_wheel_ = state.lift_wheel_;
  snapshot.look_.gamma_wheel_ = state.gamma_wheel_;
  snapshot.look_.gain_wheel_ = state.gain_wheel_;
  snapshot.look_.hls_hue_adjust_table_ = state.hls_hue_adjust_table_;
  snapshot.look_.hls_lightness_adjust_table_ = state.hls_lightness_adjust_table_;
  snapshot.look_.hls_saturation_adjust_table_ = state.hls_saturation_adjust_table_;
  snapshot.look_.hls_hue_range_table_ = state.hls_hue_range_table_;
  snapshot.look_.lut_path_ = state.lut_path_;
  // Display transform
  snapshot.display_transform_.odt_ = state.odt_;
  // Geometry
  snapshot.geometry_.rotate_degrees_ = state.rotate_degrees_;
  snapshot.geometry_.crop_enabled_ = state.crop_enabled_;
  snapshot.geometry_.crop_x_ = state.crop_x_;
  snapshot.geometry_.crop_y_ = state.crop_y_;
  snapshot.geometry_.crop_w_ = state.crop_w_;
  snapshot.geometry_.crop_h_ = state.crop_h_;
  snapshot.geometry_.crop_expand_to_fit_ = state.crop_expand_to_fit_;
  snapshot.geometry_.crop_aspect_preset_ = state.crop_aspect_preset_;
  snapshot.geometry_.crop_aspect_width_ = state.crop_aspect_width_;
  snapshot.geometry_.crop_aspect_height_ = state.crop_aspect_height_;
  // Raw decode
  snapshot.raw_.raw_highlights_reconstruct_ = state.raw_highlights_reconstruct_;
  snapshot.raw_.lens_calib_enabled_ = state.lens_calib_enabled_;
  snapshot.raw_.lens_override_make_ = state.lens_override_make_;
  snapshot.raw_.lens_override_model_ = state.lens_override_model_;
  // Render type
  snapshot.type_ = state.type_;
  return snapshot;
}

}  // namespace alcedo::ui
