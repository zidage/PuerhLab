//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/state/tone_adjustment_state.hpp"

namespace alcedo::ui {

struct TonePipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const ToneAdjustmentState& base)
      -> PipelineLoadResult<ToneAdjustmentState> {
    AdjustmentState legacy_base{};
    CopyToneStateToLegacy(base, legacy_base);
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {ToneStateFromLegacy(loaded_legacy), has_any};
  }

  static auto ParamsFor(AdjustmentField field, const ToneAdjustmentState& state) -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.exposure_     = state.exposure_;
    legacy.contrast_     = state.contrast_;
    legacy.blacks_       = state.blacks_;
    legacy.whites_       = state.whites_;
    legacy.shadows_      = state.shadows_;
    legacy.highlights_   = state.highlights_;
    legacy.curve_points_ = state.curve_points_;
    legacy.saturation_   = state.saturation_;
    legacy.sharpen_      = state.sharpen_;
    legacy.clarity_      = state.clarity_;
    return pipeline_io::ParamsForField(field, legacy, nullptr);
  }

  static auto FieldChanged(AdjustmentField field, const ToneAdjustmentState& current,
                           const ToneAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.exposure_     = current.exposure_;
    legacy_current.contrast_     = current.contrast_;
    legacy_current.blacks_       = current.blacks_;
    legacy_current.whites_       = current.whites_;
    legacy_current.shadows_      = current.shadows_;
    legacy_current.highlights_   = current.highlights_;
    legacy_current.curve_points_ = current.curve_points_;
    legacy_current.saturation_   = current.saturation_;
    legacy_current.sharpen_      = current.sharpen_;
    legacy_current.clarity_      = current.clarity_;

    AdjustmentState legacy_committed{};
    legacy_committed.exposure_     = committed.exposure_;
    legacy_committed.contrast_     = committed.contrast_;
    legacy_committed.blacks_       = committed.blacks_;
    legacy_committed.whites_       = committed.whites_;
    legacy_committed.shadows_      = committed.shadows_;
    legacy_committed.highlights_   = committed.highlights_;
    legacy_committed.curve_points_ = committed.curve_points_;
    legacy_committed.saturation_   = committed.saturation_;
    legacy_committed.sharpen_      = committed.sharpen_;
    legacy_committed.clarity_      = committed.clarity_;

    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }

 private:
  static void CopyToneStateToLegacy(const ToneAdjustmentState& tone, AdjustmentState& legacy) {
    legacy.exposure_     = tone.exposure_;
    legacy.contrast_     = tone.contrast_;
    legacy.blacks_       = tone.blacks_;
    legacy.whites_       = tone.whites_;
    legacy.shadows_      = tone.shadows_;
    legacy.highlights_   = tone.highlights_;
    legacy.curve_points_ = tone.curve_points_;
    legacy.saturation_   = tone.saturation_;
    legacy.sharpen_      = tone.sharpen_;
    legacy.clarity_      = tone.clarity_;
  }

  static auto ToneStateFromLegacy(const AdjustmentState& legacy) -> ToneAdjustmentState {
    ToneAdjustmentState tone;
    tone.exposure_     = legacy.exposure_;
    tone.contrast_     = legacy.contrast_;
    tone.blacks_       = legacy.blacks_;
    tone.whites_       = legacy.whites_;
    tone.shadows_      = legacy.shadows_;
    tone.highlights_   = legacy.highlights_;
    tone.curve_points_ = legacy.curve_points_;
    tone.saturation_   = legacy.saturation_;
    tone.sharpen_      = legacy.sharpen_;
    tone.clarity_      = legacy.clarity_;
    return tone;
  }
};

}  // namespace alcedo::ui
