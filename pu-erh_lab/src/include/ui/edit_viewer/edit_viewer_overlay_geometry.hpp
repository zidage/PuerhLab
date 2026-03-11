//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <array>
#include <optional>

#include <QPointF>
#include <QRectF>
#include <Qt>

#include "ui/edit_viewer/crop_geometry.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

namespace puerhlab {

struct EditViewerOverlaySnapshot {
  ViewerStateSnapshot   viewer_state{};
  ViewportWidgetInfo    widget_info{};
  ViewportImageInfo     image_info{};
  FramePresentationMode presentation_mode = FramePresentationMode::ViewportTransformed;
};

struct CropOverlayWidgetGeometry {
  QRectF                 image_rect{};
  bool                   image_rect_valid    = false;
  std::array<QPointF, 4> crop_corners_widget{};
  bool                   crop_corners_valid  = false;
  QPointF                rotate_stem_widget{};
  QPointF                rotate_handle_widget{};
};

enum class EditViewerOverlayHitKind {
  None,
  OutsideImage,
  BlankInImage,
  InsideCrop,
  Edge,
  Corner,
  RotateHandle,
};

struct EditViewerOverlayHover {
  EditViewerOverlayHitKind kind         = EditViewerOverlayHitKind::None;
  CropHitTestResult        crop_hit{};
  std::optional<QPointF>   image_uv{};
  bool                     inside_image = false;
  std::optional<Qt::CursorShape> cursor{};
};

class EditViewerOverlayGeometry {
 public:
  static auto Build(const EditViewerOverlaySnapshot& snapshot) -> CropOverlayWidgetGeometry;

  static auto ComputeHover(const EditViewerOverlaySnapshot& snapshot,
                           const CropOverlayWidgetGeometry& geometry, const QPointF& event_pos)
      -> EditViewerOverlayHover;
};

}  // namespace puerhlab
