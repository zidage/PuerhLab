//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>

#include "edit/pipeline/pipeline_cpu.hpp"
#include "ui/alcedo_main/editor_dialog/modules/pipeline_io.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/adjustment_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/state/geometry_adjustment_state.hpp"

namespace alcedo::ui {

struct GeometryPipelineAdapter {
  static auto Load(CPUPipelineExecutor& exec, const GeometryAdjustmentState& base)
      -> PipelineLoadResult<GeometryAdjustmentState> {
    AdjustmentState legacy_base{};
    CopyGeometryStateToLegacy(base, legacy_base);
    auto [loaded_legacy, has_any] = pipeline_io::LoadStateFromPipeline(exec, legacy_base);
    return {GeometryStateFromLegacy(loaded_legacy), has_any};
  }

  static auto ParamsFor(AdjustmentField field, const GeometryAdjustmentState& state)
      -> nlohmann::json {
    AdjustmentState legacy{};
    legacy.rotate_degrees_     = state.rotate_degrees_;
    legacy.crop_enabled_       = state.crop_enabled_;
    legacy.crop_x_             = state.crop_x_;
    legacy.crop_y_             = state.crop_y_;
    legacy.crop_w_             = state.crop_w_;
    legacy.crop_h_             = state.crop_h_;
    legacy.crop_expand_to_fit_ = state.crop_expand_to_fit_;
    legacy.crop_aspect_preset_ = state.crop_aspect_preset_;
    legacy.crop_aspect_width_  = state.crop_aspect_width_;
    legacy.crop_aspect_height_ = state.crop_aspect_height_;
    return pipeline_io::ParamsForField(field, legacy, nullptr);
  }

  static auto FieldChanged(AdjustmentField field, const GeometryAdjustmentState& current,
                           const GeometryAdjustmentState& committed) -> bool {
    AdjustmentState legacy_current{};
    legacy_current.rotate_degrees_     = current.rotate_degrees_;
    legacy_current.crop_enabled_       = current.crop_enabled_;
    legacy_current.crop_x_             = current.crop_x_;
    legacy_current.crop_y_             = current.crop_y_;
    legacy_current.crop_w_             = current.crop_w_;
    legacy_current.crop_h_             = current.crop_h_;
    legacy_current.crop_expand_to_fit_ = current.crop_expand_to_fit_;
    legacy_current.crop_aspect_preset_ = current.crop_aspect_preset_;
    legacy_current.crop_aspect_width_  = current.crop_aspect_width_;
    legacy_current.crop_aspect_height_ = current.crop_aspect_height_;

    AdjustmentState legacy_committed{};
    legacy_committed.rotate_degrees_     = committed.rotate_degrees_;
    legacy_committed.crop_enabled_       = committed.crop_enabled_;
    legacy_committed.crop_x_             = committed.crop_x_;
    legacy_committed.crop_y_             = committed.crop_y_;
    legacy_committed.crop_w_             = committed.crop_w_;
    legacy_committed.crop_h_             = committed.crop_h_;
    legacy_committed.crop_expand_to_fit_ = committed.crop_expand_to_fit_;
    legacy_committed.crop_aspect_preset_ = committed.crop_aspect_preset_;
    legacy_committed.crop_aspect_width_  = committed.crop_aspect_width_;
    legacy_committed.crop_aspect_height_ = committed.crop_aspect_height_;

    return pipeline_io::FieldChanged(field, legacy_current, legacy_committed);
  }

 private:
  static void CopyGeometryStateToLegacy(const GeometryAdjustmentState& geometry,
                                        AdjustmentState&               legacy) {
    legacy.rotate_degrees_     = geometry.rotate_degrees_;
    legacy.crop_enabled_       = geometry.crop_enabled_;
    legacy.crop_x_             = geometry.crop_x_;
    legacy.crop_y_             = geometry.crop_y_;
    legacy.crop_w_             = geometry.crop_w_;
    legacy.crop_h_             = geometry.crop_h_;
    legacy.crop_expand_to_fit_ = geometry.crop_expand_to_fit_;
    legacy.crop_aspect_preset_ = geometry.crop_aspect_preset_;
    legacy.crop_aspect_width_  = geometry.crop_aspect_width_;
    legacy.crop_aspect_height_ = geometry.crop_aspect_height_;
  }

  static auto GeometryStateFromLegacy(const AdjustmentState& legacy) -> GeometryAdjustmentState {
    GeometryAdjustmentState geometry;
    geometry.rotate_degrees_     = legacy.rotate_degrees_;
    geometry.crop_enabled_       = legacy.crop_enabled_;
    geometry.crop_x_             = legacy.crop_x_;
    geometry.crop_y_             = legacy.crop_y_;
    geometry.crop_w_             = legacy.crop_w_;
    geometry.crop_h_             = legacy.crop_h_;
    geometry.crop_expand_to_fit_ = legacy.crop_expand_to_fit_;
    geometry.crop_aspect_preset_ = legacy.crop_aspect_preset_;
    geometry.crop_aspect_width_  = legacy.crop_aspect_width_;
    geometry.crop_aspect_height_ = legacy.crop_aspect_height_;
    return geometry;
  }
};

}  // namespace alcedo::ui
