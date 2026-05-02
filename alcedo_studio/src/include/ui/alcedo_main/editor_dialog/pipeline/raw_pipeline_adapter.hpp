//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/session/adjustment_snapshot.hpp"
#include "ui/alcedo_main/editor_dialog/state/raw_decode_adjustment_state.hpp"

namespace alcedo::ui {

struct RawPipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const RawDecodeAdjustmentState& base)
      -> PipelineLoadResult<RawDecodeAdjustmentState> {
    AdjustmentState legacy_base = ToLegacyAdjustmentState(EditorAdjustmentSnapshot{.raw_ = base});
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {FromLegacyAdjustmentState(loaded_legacy).raw_, has_any};
  }

  static auto ParamsFor(AdjustmentField field, const RawDecodeAdjustmentState& state,
                        CPUPipelineExecutor* exec = nullptr) -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.raw_highlights_reconstruct_ = state.raw_highlights_reconstruct_;
    legacy.lens_calib_enabled_ = state.lens_calib_enabled_;
    legacy.lens_override_make_ = state.lens_override_make_;
    legacy.lens_override_model_ = state.lens_override_model_;
    return pipeline_io::ParamsForField(field, legacy, exec);
  }

  static auto FieldChanged(AdjustmentField field,
                           const RawDecodeAdjustmentState& current,
                           const RawDecodeAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.raw_highlights_reconstruct_ = current.raw_highlights_reconstruct_;
    legacy_current.lens_calib_enabled_ = current.lens_calib_enabled_;
    legacy_current.lens_override_make_ = current.lens_override_make_;
    legacy_current.lens_override_model_ = current.lens_override_model_;

    AdjustmentState legacy_committed{};
    legacy_committed.raw_highlights_reconstruct_ = committed.raw_highlights_reconstruct_;
    legacy_committed.lens_calib_enabled_ = committed.lens_calib_enabled_;
    legacy_committed.lens_override_make_ = committed.lens_override_make_;
    legacy_committed.lens_override_model_ = committed.lens_override_model_;

    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }
};

}  // namespace alcedo::ui
