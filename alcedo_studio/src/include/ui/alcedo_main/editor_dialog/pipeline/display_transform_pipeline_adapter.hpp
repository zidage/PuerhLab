//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/state/display_transform_adjustment_state.hpp"

namespace alcedo::ui {

struct DisplayTransformPipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const DisplayTransformAdjustmentState& base)
      -> PipelineLoadResult<DisplayTransformAdjustmentState> {
    AdjustmentState legacy_base{};
    legacy_base.odt_              = base.odt_;
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {DisplayTransformAdjustmentState{.odt_ = loaded_legacy.odt_}, has_any};
  }

  static auto ParamsFor(AdjustmentField field, const DisplayTransformAdjustmentState& state,
                        CPUPipelineExecutor* exec = nullptr) -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.odt_ = state.odt_;
    return pipeline_io::ParamsForField(field, legacy, exec);
  }

  static auto FieldChanged(AdjustmentField field, const DisplayTransformAdjustmentState& current,
                           const DisplayTransformAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.odt_ = current.odt_;
    AdjustmentState legacy_committed{};
    legacy_committed.odt_ = committed.odt_;
    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }
};

}  // namespace alcedo::ui
