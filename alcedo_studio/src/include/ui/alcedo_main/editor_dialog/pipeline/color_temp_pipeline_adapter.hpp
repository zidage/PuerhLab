//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/state/color_temp_adjustment_state.hpp"

namespace alcedo::ui {

struct ColorTempPipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const ColorTempAdjustmentState& base)
      -> PipelineLoadResult<ColorTempAdjustmentState> {
    AdjustmentState legacy_base{};
    CopyColorTempStateToLegacy(base, legacy_base);
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {ColorTempStateFromLegacy(loaded_legacy), has_any};
  }

  static auto ParamsFor(AdjustmentField field, const ColorTempAdjustmentState& state)
      -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.color_temp_mode_          = state.mode_;
    legacy.color_temp_custom_cct_    = state.custom_cct_;
    legacy.color_temp_custom_tint_   = state.custom_tint_;
    legacy.color_temp_resolved_cct_  = state.resolved_cct_;
    legacy.color_temp_resolved_tint_ = state.resolved_tint_;
    return pipeline_io::ParamsForField(field, legacy, nullptr);
  }

  static auto FieldChanged(AdjustmentField field, const ColorTempAdjustmentState& current,
                           const ColorTempAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.color_temp_mode_          = current.mode_;
    legacy_current.color_temp_custom_cct_    = current.custom_cct_;
    legacy_current.color_temp_custom_tint_   = current.custom_tint_;
    legacy_current.color_temp_resolved_cct_  = current.resolved_cct_;
    legacy_current.color_temp_resolved_tint_ = current.resolved_tint_;

    AdjustmentState legacy_committed{};
    legacy_committed.color_temp_mode_          = committed.mode_;
    legacy_committed.color_temp_custom_cct_    = committed.custom_cct_;
    legacy_committed.color_temp_custom_tint_   = committed.custom_tint_;
    legacy_committed.color_temp_resolved_cct_  = committed.resolved_cct_;
    legacy_committed.color_temp_resolved_tint_ = committed.resolved_tint_;

    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }

 private:
  static void CopyColorTempStateToLegacy(const ColorTempAdjustmentState& color_temp,
                                         AdjustmentState&                legacy) {
    legacy.color_temp_mode_          = color_temp.mode_;
    legacy.color_temp_custom_cct_    = color_temp.custom_cct_;
    legacy.color_temp_custom_tint_   = color_temp.custom_tint_;
    legacy.color_temp_resolved_cct_  = color_temp.resolved_cct_;
    legacy.color_temp_resolved_tint_ = color_temp.resolved_tint_;
    legacy.color_temp_supported_     = color_temp.supported_;
  }

  static auto ColorTempStateFromLegacy(const AdjustmentState& legacy) -> ColorTempAdjustmentState {
    ColorTempAdjustmentState color_temp;
    color_temp.mode_          = legacy.color_temp_mode_;
    color_temp.custom_cct_    = legacy.color_temp_custom_cct_;
    color_temp.custom_tint_   = legacy.color_temp_custom_tint_;
    color_temp.resolved_cct_  = legacy.color_temp_resolved_cct_;
    color_temp.resolved_tint_ = legacy.color_temp_resolved_tint_;
    color_temp.supported_     = legacy.color_temp_supported_;
    return color_temp;
  }
};

}  // namespace alcedo::ui
