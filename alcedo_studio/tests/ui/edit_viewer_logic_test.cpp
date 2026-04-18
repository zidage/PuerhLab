//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include "ui/edit_viewer/crop_geometry.hpp"
#include "ui/edit_viewer/crop_interaction_controller.hpp"
#include "ui/edit_viewer/view_transform_controller.hpp"
#include "ui/edit_viewer/viewer_state.hpp"
#include "ui/edit_viewer/viewport_mapper.hpp"

namespace alcedo {
namespace {

const ViewportWidgetInfo kWidgetInfo{800, 600, 1.0f};
const ViewportImageInfo  kImageInfo{400, 300};

auto WidgetPointForUv(const QPointF& uv) -> QPointF {
  const auto point = ViewportMapper::ImageUvToWidgetPoint(uv, kWidgetInfo, kImageInfo, 1.0f,
                                                          QVector2D(0.0f, 0.0f));
  EXPECT_TRUE(point.has_value());
  return point.value_or(QPointF());
}

}  // namespace

TEST(EditViewerLogicTests, ViewportMapperRoundTripsUvThroughWidgetSpace) {
  const QPointF uv(0.23, 0.67);
  const auto    widget_point =
      ViewportMapper::ImageUvToWidgetPoint(uv, kWidgetInfo, kImageInfo, 1.8f, QVector2D(0.1f, -0.2f));
  ASSERT_TRUE(widget_point.has_value());

  const auto round_trip = ViewportMapper::WidgetPointToImageUv(*widget_point, kWidgetInfo, kImageInfo,
                                                               1.8f, QVector2D(0.1f, -0.2f));
  ASSERT_TRUE(round_trip.has_value());
  EXPECT_NEAR(round_trip->x(), uv.x(), 1e-5);
  EXPECT_NEAR(round_trip->y(), uv.y(), 1e-5);
}

TEST(EditViewerLogicTests, CropGeometryAspectLockedDiagonalPreservesRatio) {
  const QRectF rect = CropGeometry::MakeAspectLockedRectFromDiagonal(QPointF(0.2, 0.2),
                                                                     QPointF(0.7, 0.5), 4.0f / 3.0f,
                                                                     16.0f / 9.0f);
  ASSERT_GT(rect.width(), 0.0);
  ASSERT_GT(rect.height(), 0.0);
  EXPECT_NEAR((static_cast<float>(rect.width()) * (4.0f / 3.0f)) / static_cast<float>(rect.height()),
              16.0f / 9.0f, 1e-4f);
}

TEST(EditViewerLogicTests, CropGeometryRotationClampKeepsCornersInsideNormalizedImage) {
  const QRectF rect = CropGeometry::ClampCropRectForRotation(QRectF(0.0, 0.0, 1.0, 1.0), 37.0f, 4.0f / 3.0f);
  const auto   corners = CropGeometry::RotatedCropCornersUv(rect, 37.0f, 4.0f / 3.0f);
  for (const auto& corner : corners) {
    EXPECT_GE(corner.x(), -1e-5);
    EXPECT_LE(corner.x(), 1.0 + 1e-5);
    EXPECT_GE(corner.y(), -1e-5);
    EXPECT_LE(corner.y(), 1.0 + 1e-5);
  }
}

TEST(EditViewerLogicTests, ViewTransformControllerCtrlWheelUpdatesZoomAndPan) {
  ViewerState             state;
  ViewTransformController controller;

  const auto result = controller.HandleCtrlWheel(state, kWidgetInfo, kImageInfo, 120,
                                                 QPointF(400.0, 300.0));
  EXPECT_TRUE(result.consumed);
  EXPECT_TRUE(result.request_repaint);
  ASSERT_TRUE(result.emitted_zoom.has_value());
  EXPECT_GT(*result.emitted_zoom, 1.0f);
  EXPECT_GT(state.GetViewZoom(), 1.0f);
}

TEST(EditViewerLogicTests, ViewTransformControllerDoubleClickStartsAnimationAndReturnsToFit) {
  ViewerState             state;
  ViewTransformController controller;

  const auto zoom_in = controller.HandleDoubleClick(state, kWidgetInfo, kImageInfo, QPointF(400.0, 300.0));
  EXPECT_TRUE(zoom_in.start_animation);
  const auto progress = controller.ApplyAnimationFinished(state, kWidgetInfo, kImageInfo);
  ASSERT_TRUE(progress.emitted_zoom.has_value());
  EXPECT_GT(*progress.emitted_zoom, 1.0f);

  const auto zoom_out = controller.HandleDoubleClick(state, kWidgetInfo, kImageInfo, QPointF(400.0, 300.0));
  EXPECT_TRUE(zoom_out.start_animation);
  const auto finished = controller.ApplyAnimationFinished(state, kWidgetInfo, kImageInfo);
  ASSERT_TRUE(finished.emitted_zoom.has_value());
  EXPECT_FLOAT_EQ(*finished.emitted_zoom, 1.0f);
}

TEST(EditViewerLogicTests, CropInteractionControllerCreatesAndFinalizesCropRect) {
  ViewerState                state;
  CropInteractionController  controller;
  auto                       crop_state = state.GetCropOverlay();
  crop_state.tool_enabled    = true;
  crop_state.overlay_visible = true;
  crop_state.rect            = QRectF(0.25, 0.25, 0.5, 0.5);
  state.SetCropOverlayState(crop_state);

  const QPointF start_point = WidgetPointForUv(QPointF(0.1, 0.1));
  const auto    press = controller.HandlePress(state, kWidgetInfo, kImageInfo, start_point);
  EXPECT_TRUE(press.consumed);
  EXPECT_TRUE(press.rect_changed.has_value());

  const QPointF end_point = WidgetPointForUv(QPointF(0.4, 0.45));
  const auto    move = controller.HandleMove(state, kWidgetInfo, kImageInfo, Qt::LeftButton, end_point);
  EXPECT_TRUE(move.consumed);
  ASSERT_TRUE(move.rect_changed.has_value());
  EXPECT_NEAR(move.rect_changed->x(), 0.1, 1e-3);
  EXPECT_NEAR(move.rect_changed->y(), 0.1, 1e-3);
  EXPECT_GT(move.rect_changed->width(), 0.25);
  EXPECT_GT(move.rect_changed->height(), 0.3);

  const auto release = controller.HandleRelease(state);
  EXPECT_TRUE(release.consumed);
  EXPECT_TRUE(release.rect_is_final);
}

TEST(EditViewerLogicTests, CropGeometryHitTestPrefersCornersOverEdges) {
  const QRectF rect = QRectF(0.2, 0.2, 0.4, 0.4);
  const auto   corners_uv =
      CropGeometry::RotatedCropCornersUv(rect, 0.0f, CropGeometry::SafeAspect(kImageInfo.image_width,
                                                                               kImageInfo.image_height));
  std::array<QPointF, 4> corners_widget{};
  for (size_t i = 0; i < corners_uv.size(); ++i) {
    const auto point = ViewportMapper::ImageUvToWidgetPoint(corners_uv[i], kWidgetInfo, kImageInfo,
                                                            1.0f, QVector2D(0.0f, 0.0f));
    ASSERT_TRUE(point.has_value());
    corners_widget[i] = *point;
  }

  const auto hit = CropGeometry::HitTestWidgetGeometry(corners_widget, corners_widget[0]);
  EXPECT_EQ(hit.corner_index, 0);
  EXPECT_EQ(hit.edge, CropEdge::None);
  EXPECT_FALSE(hit.rotate_handle_hit);
}

}  // namespace alcedo
