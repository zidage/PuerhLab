//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include <gtest/gtest.h>

#include <algorithm>

#include <QApplication>
#include <QMouseEvent>
#include <QPointF>

#include "ui/alcedo_main/editor_dialog/modules/curve.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_curve_widget.hpp"

namespace alcedo::ui {
namespace {

auto PlotRectForSize(int width, int height) -> QRectF {
  constexpr qreal kLeft   = 22.0;
  constexpr qreal kTop    = 14.0;
  constexpr qreal kRight  = 12.0;
  constexpr qreal kBottom = 24.0;
  return QRectF(kLeft, kTop, std::max(30.0, width - kLeft - kRight),
                std::max(30.0, height - kTop - kBottom));
}

auto ToWidgetPoint(const QPointF& normalized, int width, int height) -> QPointF {
  const QRectF plot = PlotRectForSize(width, height);
  const qreal  x    = plot.left() + normalized.x() * plot.width();
  const qreal  y    = plot.bottom() - normalized.y() * plot.height();
  return QPointF(x, y);
}

void SendMousePress(QWidget& widget, const QPointF& pos) {
  QMouseEvent event(QEvent::MouseButtonPress, pos, pos, pos, Qt::LeftButton, Qt::LeftButton,
                    Qt::NoModifier);
  QApplication::sendEvent(&widget, &event);
}

void SendMouseMove(QWidget& widget, const QPointF& pos) {
  QMouseEvent event(QEvent::MouseMove, pos, pos, pos, Qt::NoButton, Qt::LeftButton,
                    Qt::NoModifier);
  QApplication::sendEvent(&widget, &event);
}

void SendMouseRelease(QWidget& widget, const QPointF& pos) {
  QMouseEvent event(QEvent::MouseButtonRelease, pos, pos, pos, Qt::LeftButton, Qt::NoButton,
                    Qt::NoModifier);
  QApplication::sendEvent(&widget, &event);
}

}  // namespace

TEST(ToneCurveWidgetTests, NormalizeCurveControlPointsKeepsMovedEndpoints) {
  const auto normalized = curve::NormalizeCurveControlPoints(
      {QPointF(0.15, 0.10), QPointF(0.45, 0.40), QPointF(0.85, 0.92)});

  ASSERT_EQ(normalized.size(), 3u);
  EXPECT_NEAR(normalized.front().x(), 0.15, 1e-6);
  EXPECT_NEAR(normalized.front().y(), 0.10, 1e-6);
  EXPECT_NEAR(normalized.back().x(), 0.85, 1e-6);
  EXPECT_NEAR(normalized.back().y(), 0.92, 1e-6);
}

TEST(ToneCurveWidgetTests, CurveOpEndpointMovesUseExpectedToneCurveSemantics) {
  const std::vector<QPointF> lifted_black = {
      QPointF(0.0, 0.2), QPointF(0.25, 0.25), QPointF(0.75, 0.75), QPointF(1.0, 1.0)};
  EXPECT_GT(curve::EvaluateCurveHermite(0.10f, lifted_black, curve::BuildCurveHermiteCache(lifted_black)),
            0.10f);

  const std::vector<QPointF> lowered_white = {
      QPointF(0.0, 0.0), QPointF(0.25, 0.25), QPointF(0.75, 0.75), QPointF(1.0, 0.8)};
  EXPECT_LT(curve::EvaluateCurveHermite(0.90f, lowered_white, curve::BuildCurveHermiteCache(lowered_white)),
            0.90f);

  const std::vector<QPointF> black_point_right = {
      QPointF(0.2, 0.0), QPointF(0.25, 0.25), QPointF(0.75, 0.75), QPointF(1.0, 1.0)};
  EXPECT_NEAR(
      curve::EvaluateCurveHermite(0.10f, black_point_right, curve::BuildCurveHermiteCache(black_point_right)),
      0.0f, 1e-6f);

  const std::vector<QPointF> white_point_left = {
      QPointF(0.0, 0.0), QPointF(0.25, 0.25), QPointF(0.75, 0.75), QPointF(0.8, 1.0)};
  EXPECT_NEAR(
      curve::EvaluateCurveHermite(0.90f, white_point_left, curve::BuildCurveHermiteCache(white_point_left)),
      1.0f, 1e-6f);
}

TEST(ToneCurveWidgetTests, EndpointHandlesCanMoveHorizontally) {
  ToneCurveWidget widget;
  widget.resize(320, 240);
  widget.show();
  QApplication::processEvents();

  const QPointF black_start = ToWidgetPoint(QPointF(0.0, 0.0), widget.width(), widget.height());
  const QPointF black_end   = ToWidgetPoint(QPointF(0.18, 0.22), widget.width(), widget.height());
  SendMousePress(widget, black_start);
  SendMouseMove(widget, black_end);
  SendMouseRelease(widget, black_end);

  ASSERT_FALSE(widget.GetControlPoints().empty());
  EXPECT_NEAR(widget.GetControlPoints().front().x(), 0.18, 0.02);
  EXPECT_NEAR(widget.GetControlPoints().front().y(), 0.22, 0.02);

  const auto&   after_black = widget.GetControlPoints();
  const QPointF white_start =
      ToWidgetPoint(after_black.back(), widget.width(), widget.height());
  const QPointF white_end   = ToWidgetPoint(QPointF(0.82, 0.78), widget.width(), widget.height());
  SendMousePress(widget, white_start);
  SendMouseMove(widget, white_end);
  SendMouseRelease(widget, white_end);

  ASSERT_GE(widget.GetControlPoints().size(), 2u);
  EXPECT_NEAR(widget.GetControlPoints().back().x(), 0.82, 0.02);
  EXPECT_NEAR(widget.GetControlPoints().back().y(), 0.78, 0.02);
}

}  // namespace alcedo::ui
