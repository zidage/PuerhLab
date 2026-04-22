//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <cmath>

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

TEST(EditViewerLogicTests, ViewportRenderRegionCarriesReferenceSizeForDetailPreviewRequests) {
  const auto region = ViewportMapper::ComputeViewportRenderRegion(
      kWidgetInfo, 2.0f, QVector2D(0.15f, -0.1f), 4096, 3072);
  ASSERT_TRUE(region.has_value());
  EXPECT_EQ(region->reference_width_, 4096);
  EXPECT_EQ(region->reference_height_, 3072);
  EXPECT_LT(region->scale_x_, 1.0f);
  EXPECT_LT(region->scale_y_, 1.0f);
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

TEST(EditViewerLogicTests, AnchoredPanMustUseReferenceImageGeometryForViewportRoiViews) {
  const ViewportWidgetInfo widget_info{800, 600, 1.0f};
  const ViewportImageInfo  reference_image{6000, 4000};
  const QVector2D          current_pan(0.6f, 0.1f);
  constexpr float          current_zoom = 1.5f;
  constexpr float          target_zoom  = 2.0f;
  const QPointF            anchor_widget_pos(100.0, 500.0);

  const auto viewport_region = ViewportMapper::ComputeViewportRenderRegion(
      widget_info, current_zoom, current_pan, reference_image.image_width,
      reference_image.image_height);
  ASSERT_TRUE(viewport_region.has_value());

  const ViewportImageInfo roi_image{
      std::max(1, static_cast<int>(std::lround(static_cast<double>(reference_image.image_width) *
                                               viewport_region->scale_x_))),
      std::max(1, static_cast<int>(std::lround(static_cast<double>(reference_image.image_height) *
                                               viewport_region->scale_y_)))};

  const auto anchor_uv_before = ViewportMapper::WidgetPointToImageUv(
      anchor_widget_pos, widget_info, reference_image, current_zoom, current_pan);
  ASSERT_TRUE(anchor_uv_before.has_value());

  const auto unclamped_reference_pan =
      ViewportMapper::ComputeAnchoredPan(anchor_widget_pos, widget_info, reference_image,
                                         current_zoom, current_pan, target_zoom, current_pan);
  const auto reference_pan =
      ViewportMapper::ClampPanForZoom(widget_info, reference_image, target_zoom,
                                      unclamped_reference_pan, 1.0f, 8.0f);
  const auto anchored_uv_with_reference = ViewportMapper::WidgetPointToImageUv(
      anchor_widget_pos, widget_info, reference_image, target_zoom, reference_pan);
  ASSERT_TRUE(anchored_uv_with_reference.has_value());
  EXPECT_LT(std::abs(anchored_uv_with_reference->x() - anchor_uv_before->x()), 2.0e-2);
  EXPECT_NEAR(anchored_uv_with_reference->y(), anchor_uv_before->y(), 1.0e-5);

  const auto unclamped_roi_pan =
      ViewportMapper::ComputeAnchoredPan(anchor_widget_pos, widget_info, roi_image, current_zoom,
                                         current_pan, target_zoom, current_pan);
  const auto roi_pan = ViewportMapper::ClampPanForZoom(widget_info, roi_image, target_zoom,
                                                       unclamped_roi_pan, 1.0f, 8.0f);
  const auto anchored_uv_with_roi = ViewportMapper::WidgetPointToImageUv(
      anchor_widget_pos, widget_info, reference_image, target_zoom, roi_pan);
  ASSERT_TRUE(anchored_uv_with_roi.has_value());
  EXPECT_GT(std::abs(roi_pan.x() - reference_pan.x()), 5.0e-2);
  EXPECT_GT(std::abs(anchored_uv_with_roi->x() - anchor_uv_before->x()),
            std::abs(anchored_uv_with_reference->x() - anchor_uv_before->x()));
  EXPECT_NEAR(anchored_uv_with_roi->y(), anchor_uv_before->y(), 1.0e-5);
}

TEST(EditViewerLogicTests, CropAspectChangesRequirePanToBeReclampedToNewReference) {
  const ViewportWidgetInfo widget_info{800, 600, 1.0f};
  const ViewportImageInfo  pre_crop_image{4000, 3000};
  const ViewportImageInfo  post_crop_image{3000, 3000};
  constexpr float          zoom = 2.0f;

  const QVector2D pre_crop_pan =
      ViewportMapper::ClampPanForZoom(widget_info, pre_crop_image, zoom, QVector2D(0.7f, 0.0f),
                                      ViewTransformController::kMinInteractiveZoom,
                                      ViewTransformController::kMaxInteractiveZoom);
  const QVector2D post_crop_pan = ViewportMapper::ClampPanForZoom(
      widget_info, post_crop_image, zoom, pre_crop_pan,
      ViewTransformController::kMinInteractiveZoom,
      ViewTransformController::kMaxInteractiveZoom);

  EXPECT_GT(pre_crop_pan.x(), post_crop_pan.x());
  EXPECT_FLOAT_EQ(post_crop_pan.y(), 0.0f);
  EXPECT_LE(post_crop_pan.x(), 0.5f);
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
