//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/session/adjustment_snapshot.hpp"
#include "ui/alcedo_main/editor_dialog/state/look_adjustment_state.hpp"

namespace alcedo::ui {

struct LookPipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const LookAdjustmentState& base)
      -> PipelineLoadResult<LookAdjustmentState> {
    AdjustmentState legacy_base = ToLegacyAdjustmentState(EditorAdjustmentSnapshot{.look_ = base});
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {FromLegacyAdjustmentState(loaded_legacy).look_, has_any};
  }

  static auto ParamsFor(AdjustmentField field, const LookAdjustmentState& state) -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.hls_target_hue_ = state.hls_target_hue_;
    legacy.hls_hue_adjust_ = state.hls_hue_adjust_;
    legacy.hls_lightness_adjust_ = state.hls_lightness_adjust_;
    legacy.hls_saturation_adjust_ = state.hls_saturation_adjust_;
    legacy.hls_hue_range_ = state.hls_hue_range_;
    legacy.lift_wheel_ = state.lift_wheel_;
    legacy.gamma_wheel_ = state.gamma_wheel_;
    legacy.gain_wheel_ = state.gain_wheel_;
    legacy.hls_hue_adjust_table_ = state.hls_hue_adjust_table_;
    legacy.hls_lightness_adjust_table_ = state.hls_lightness_adjust_table_;
    legacy.hls_saturation_adjust_table_ = state.hls_saturation_adjust_table_;
    legacy.hls_hue_range_table_ = state.hls_hue_range_table_;
    legacy.lut_path_ = state.lut_path_;
    return pipeline_io::ParamsForField(field, legacy, nullptr);
  }

  static auto FieldChanged(AdjustmentField field,
                           const LookAdjustmentState& current,
                           const LookAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.hls_target_hue_ = current.hls_target_hue_;
    legacy_current.hls_hue_adjust_ = current.hls_hue_adjust_;
    legacy_current.hls_lightness_adjust_ = current.hls_lightness_adjust_;
    legacy_current.hls_saturation_adjust_ = current.hls_saturation_adjust_;
    legacy_current.hls_hue_range_ = current.hls_hue_range_;
    legacy_current.lift_wheel_ = current.lift_wheel_;
    legacy_current.gamma_wheel_ = current.gamma_wheel_;
    legacy_current.gain_wheel_ = current.gain_wheel_;
    legacy_current.hls_hue_adjust_table_ = current.hls_hue_adjust_table_;
    legacy_current.hls_lightness_adjust_table_ = current.hls_lightness_adjust_table_;
    legacy_current.hls_saturation_adjust_table_ = current.hls_saturation_adjust_table_;
    legacy_current.hls_hue_range_table_ = current.hls_hue_range_table_;
    legacy_current.lut_path_ = current.lut_path_;

    AdjustmentState legacy_committed{};
    legacy_committed.hls_target_hue_ = committed.hls_target_hue_;
    legacy_committed.hls_hue_adjust_ = committed.hls_hue_adjust_;
    legacy_committed.hls_lightness_adjust_ = committed.hls_lightness_adjust_;
    legacy_committed.hls_saturation_adjust_ = committed.hls_saturation_adjust_;
    legacy_committed.hls_hue_range_ = committed.hls_hue_range_;
    legacy_committed.lift_wheel_ = committed.lift_wheel_;
    legacy_committed.gamma_wheel_ = committed.gamma_wheel_;
    legacy_committed.gain_wheel_ = committed.gain_wheel_;
    legacy_committed.hls_hue_adjust_table_ = committed.hls_hue_adjust_table_;
    legacy_committed.hls_lightness_adjust_table_ = committed.hls_lightness_adjust_table_;
    legacy_committed.hls_saturation_adjust_table_ = committed.hls_saturation_adjust_table_;
    legacy_committed.hls_hue_range_table_ = committed.hls_hue_range_table_;
    legacy_committed.lut_path_ = committed.lut_path_;

    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }
};

}  // namespace alcedo::ui
