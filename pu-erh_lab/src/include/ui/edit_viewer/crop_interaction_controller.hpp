//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <optional>

#include <QPointF>
#include <QRectF>
#include <Qt>

#include "ui/edit_viewer/crop_geometry.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

namespace puerhlab {

enum class CropDragMode {
  None,
  Create,
  Move,
  ResizeEdge,
  ResizeCorner,
  RotateHandle,
};

struct CropInteractionResult {
  bool                       consumed       = false;
  bool                       request_repaint = false;
  std::optional<Qt::CursorShape> cursor{};
  bool                       unset_cursor   = false;
  std::optional<QRectF>      rect_changed{};
  bool                       rect_is_final  = false;
  std::optional<float>       rotation_changed{};
  bool                       rotation_is_final = false;
};

class CropInteractionController {
 public:
  CropInteractionController() = default;

  auto HandlePress(ViewerState& state, const ViewportWidgetInfo& widget_info,
                   const ViewportImageInfo& image_info, const QPointF& event_pos)
      -> CropInteractionResult;

  auto HandleMove(ViewerState& state, const ViewportWidgetInfo& widget_info,
                  const ViewportImageInfo& image_info, Qt::MouseButtons buttons,
                  const QPointF& event_pos) -> CropInteractionResult;

  auto HandleRelease(ViewerState& state) -> CropInteractionResult;
  auto HandleDoubleClick(ViewerState& state) -> CropInteractionResult;
  void Cancel();

 private:
  auto MakeRectEmissionResult(const QRectF& rect, bool is_final) const -> CropInteractionResult;

  CropDragMode drag_mode_               = CropDragMode::None;
  CropCorner   drag_corner_             = CropCorner::None;
  CropEdge     drag_edge_               = CropEdge::None;
  QPointF      drag_anchor_uv_{};
  QPointF      drag_anchor_widget_pos_{};
  QRectF       drag_origin_rect_{};
  QPointF      drag_fixed_corner_uv_{};
  float        drag_rotation_degrees_   = 0.0f;
};

}  // namespace puerhlab
