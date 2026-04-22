//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/editor_viewer_pane.hpp"

#include <QPainter>

namespace alcedo::ui {
namespace {

auto RenderActualPixelsIcon(const QColor& color, const QSize& size, qreal dpr) -> QIcon {
  const int w = size.width();
  const int h = size.height();
  QPixmap pixmap(QSize(w, h) * std::max(1.0, dpr));
  pixmap.setDevicePixelRatio(std::max(1.0, dpr));
  pixmap.fill(Qt::transparent);
  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing);
  QPen pen(color, 1.3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
  painter.setPen(pen);
  painter.setBrush(Qt::NoBrush);
  const qreal m  = 2.5;
  const qreal e  = 3.5;
  const qreal left   = m;
  const qreal top    = m;
  const qreal right  = w - m;
  const qreal bottom = h - m;
  painter.drawLine(QPointF(left, top + e), QPointF(left, top));
  painter.drawLine(QPointF(left, top), QPointF(left + e, top));
  painter.drawLine(QPointF(right - e, top), QPointF(right, top));
  painter.drawLine(QPointF(right, top), QPointF(right, top + e));
  painter.drawLine(QPointF(right, bottom - e), QPointF(right, bottom));
  painter.drawLine(QPointF(right, bottom), QPointF(right - e, bottom));
  painter.drawLine(QPointF(left + e, bottom), QPointF(left, bottom));
  painter.drawLine(QPointF(left, bottom), QPointF(left, bottom - e));
  return QIcon(pixmap);
}

}  // namespace

EditorViewerPane::EditorViewerPane(QWidget* parent) : QWidget(parent) {}

void EditorDialog::BuildViewerAndPanelShell() {
    const auto& theme = AppTheme::Instance();
    const QString borderless_panel_style = QStringLiteral(
        "#EditorControlsPanel {"
        "  background: %1;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorControlsPanel QFrame#EditorSection {"
        "  background: %2;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionTitle {"
        "  color: %3;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionSub {"
        "  color: %4;"
        "}")
                                          .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                               theme.bgBaseColor().name(QColor::HexArgb),
                                               theme.textColor().name(QColor::HexRgb),
                                               theme.textMutedColor().name(QColor::HexRgb));
    const QString version_panel_style = QStringLiteral(
        "#EditorVersioningPanel {"
        "  background: %1;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorVersioningPanel QFrame#EditorSection {"
        "  background: transparent;"
        "  border: none;"
        "  border-radius: 0px;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionTitle {"
        "  color: %3;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionSub {"
        "  color: %4;"
        "}")
                                           .arg(QColor(theme.bgDeepColor().red(),
                                                       theme.bgDeepColor().green(),
                                                       theme.bgDeepColor().blue(), 246)
                                                    .name(QColor::HexArgb),
                                                theme.textColor().name(QColor::HexRgb),
                                                theme.textMutedColor().name(QColor::HexRgb));
    // Sidebar buttons: solid-gray rounded-square tiles, no borders (Qt Widgets
    // renders white strokes with visible dots on this system — avoid them).
    const QColor rail_tile       = theme.bgPanelColor();
    const QString version_rail_style = QStringLiteral(
        "#EditorVersioningRail {"
        "  background: %1;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorVersioningRail QPushButton {"
        "  min-width: 40px;"
        "  min-height: 40px;"
        "  border-radius: 10px;"
        "  border: none;"
        "  background: %2;"
        "  padding: 0px;"
        "}"
        "#EditorVersioningRail QPushButton:hover {"
        "  background: %3;"
        "}"
        "#EditorVersioningRail QPushButton:pressed {"
        "  background: %4;"
        "}"
        "#EditorVersioningRail QPushButton[versioningActive=\"true\"] {"
        "  background: %5;"
        "}"
        "#EditorVersioningRail QPushButton[versioningActive=\"true\"]:hover {"
        "  background: %6;"
        "}")
                                          .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                               QColor(rail_tile.red(), rail_tile.green(),
                                                      rail_tile.blue(), 210)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_tile.red(), rail_tile.green(),
                                                      rail_tile.blue(), 238)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_tile.red(), rail_tile.green(),
                                                      rail_tile.blue(), 170)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_tile.red(), rail_tile.green(),
                                                      rail_tile.blue(), 255)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_tile.red(), rail_tile.green(),
                                                      rail_tile.blue(), 255)
                                                   .name(QColor::HexArgb));

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(kEditorOuterMargin, kEditorOuterMargin, kEditorOuterMargin,
                             kEditorOuterMargin);
    root->setSpacing(0);

    auto* main_splitter = new QSplitter(Qt::Horizontal, this);
    main_splitter->setObjectName("EditorMainSplitter");
    main_splitter->setChildrenCollapsible(false);
    main_splitter->setHandleWidth(kEditorOuterMargin);
    main_splitter->setStyleSheet(
        "QSplitter#EditorMainSplitter::handle {"
        "  background: transparent;"
        "}");
    main_splitter_ = main_splitter;

    viewer_ = new QtEditViewer(this);
    viewer_->setProperty("cornerRadius", 14);
    viewer_->setMinimumSize(420, 320);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QWidget {"
        "  background: #151A20;"
        "  border: none;"
        "  border-radius: 14px;"
        "}");

    viewer_container_ = new EditorViewerPane(this);
    viewer_container_->setObjectName("EditorViewportFrame");
    viewer_container_->setAttribute(Qt::WA_StyledBackground, true);
    viewer_container_->setStyleSheet(
        "#EditorViewportFrame {"
        "  background: #151A20;"
        "  border: none;"
        "  border-radius: 14px;"
        "}");
    auto* viewer_grid = new QGridLayout(viewer_container_);
    viewer_grid->setContentsMargins(0, 0, 0, 0);
    viewer_grid->setSpacing(0);
    viewer_grid->addWidget(viewer_, 0, 0);

    {
      auto* zoom_anchor = new QWidget(viewer_container_);
      auto* zoom_anchor_layout = new QHBoxLayout(zoom_anchor);
      zoom_anchor_layout->setContentsMargins(0, 10, 12, 0);
      zoom_anchor_layout->setSpacing(0);
      zoom_anchor_layout->addStretch(1);

      viewer_zoom_overlay_ = new QWidget(zoom_anchor);
      viewer_zoom_overlay_->setObjectName(QStringLiteral("ViewerZoomOverlay"));

      auto* overlay_layout = new QHBoxLayout(viewer_zoom_overlay_);
      overlay_layout->setContentsMargins(10, 5, 10, 5);
      overlay_layout->setSpacing(0);

      const QColor bg_color = QColor(theme.bgDeepColor().red(), theme.bgDeepColor().green(),
                                     theme.bgDeepColor().blue(), 220);
      const QColor border_color = QColor(theme.glassStrokeColor().red(),
                                         theme.glassStrokeColor().green(),
                                         theme.glassStrokeColor().blue(), 140);
      viewer_zoom_overlay_->setStyleSheet(QStringLiteral(
          "QWidget#ViewerZoomOverlay {"
          "  background: %1;"
          "  border: 1px solid %2;"
          "  border-radius: 10px;"
          "}"
          "QLabel { color: %3; background: transparent; border: none; }")
          .arg(bg_color.name(QColor::HexArgb),
               border_color.name(QColor::HexArgb),
               theme.textColor().name(QColor::HexRgb)));

      auto make_divider = [&](QWidget* parent) -> QFrame* {
        auto* divider = new QFrame(parent);
        divider->setFrameShape(QFrame::VLine);
        divider->setFrameShadow(QFrame::Plain);
        divider->setFixedWidth(1);
        divider->setFixedHeight(14);
        divider->setStyleSheet(QStringLiteral("color: %1; background: %1; border: none;")
                                   .arg(QColor(theme.dividerColor().red(),
                                               theme.dividerColor().green(),
                                               theme.dividerColor().blue(), 120)
                                            .name(QColor::HexArgb)));
        return divider;
      };

      viewer_zoom_value_label_ = new QLabel(QStringLiteral("100%"), viewer_zoom_overlay_);
      AppTheme::MarkFontRole(viewer_zoom_value_label_, AppTheme::FontRole::DataOverlay);
      overlay_layout->addWidget(viewer_zoom_value_label_, 0, Qt::AlignVCenter);

      overlay_layout->addSpacing(8);
      overlay_layout->addWidget(make_divider(viewer_zoom_overlay_), 0, Qt::AlignVCenter);
      overlay_layout->addSpacing(8);

      viewer_zoom_resolution_label_ = new QLabel(QStringLiteral("-- × -- px"), viewer_zoom_overlay_);
      AppTheme::MarkFontRole(viewer_zoom_resolution_label_, AppTheme::FontRole::DataCaption);
      viewer_zoom_resolution_label_->setStyleSheet(QStringLiteral("color: %1;")
                                                       .arg(theme.textMutedColor().name(QColor::HexRgb)));
      overlay_layout->addWidget(viewer_zoom_resolution_label_, 0, Qt::AlignVCenter);

      overlay_layout->addSpacing(8);
      overlay_layout->addWidget(make_divider(viewer_zoom_overlay_), 0, Qt::AlignVCenter);
      overlay_layout->addSpacing(6);

      auto* reset_btn = new QPushButton(viewer_zoom_overlay_);
      reset_btn->setFixedSize(18, 18);
      reset_btn->setCursor(Qt::PointingHandCursor);
      reset_btn->setFlat(true);
      reset_btn->setIconSize(QSize(16, 16));
      reset_btn->setIcon(RenderActualPixelsIcon(theme.textMutedColor(), QSize(16, 16),
                                                reset_btn->devicePixelRatioF()));
      reset_btn->setStyleSheet(QStringLiteral(
          "QPushButton { background: transparent; border: none; padding: 0px; }"
          "QPushButton:hover { background: transparent; border: none; }"));
      QObject::connect(reset_btn, &QPushButton::clicked, this, [this]() {
        if (viewer_) {
          viewer_->ResetView();
        }
      });
      overlay_layout->addWidget(reset_btn, 0, Qt::AlignVCenter);

      zoom_anchor_layout->addWidget(viewer_zoom_overlay_, 0, Qt::AlignRight | Qt::AlignVCenter);
      viewer_grid->addWidget(zoom_anchor, 0, 0, Qt::AlignRight | Qt::AlignTop);
    }

    spinner_ = new SpinnerWidget(viewer_container_);
    viewer_grid->addWidget(spinner_, 0, 0, Qt::AlignRight | Qt::AlignBottom);
    viewer_grid->setRowStretch(0, 1);
    viewer_grid->setColumnStretch(0, 1);

    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::ViewZoomChanged, this,
                       [this](float zoom) { UpdateViewerZoomLabel(zoom); });
      UpdateViewerZoomLabel(viewer_->GetViewZoom());
    }

    auto* controls_panel = BuildControlPanelShell(borderless_panel_style);

    // ── Versioning rail (fixed 64 px, always visible) ─────────────────────
    versioning_panel_host_ = new VersioningPanelWidget(this);
    versioning_panel_host_->setFixedWidth(kVersioningCollapsedWidth);
    versioning_panel_host_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    auto* versioning_host_layout = new QHBoxLayout(versioning_panel_host_);
    versioning_host_layout->setContentsMargins(0, 0, 0, 0);
    versioning_host_layout->setSpacing(0);

    versioning_collapsed_nav_ = new QWidget(versioning_panel_host_);
    versioning_collapsed_nav_->setObjectName("EditorVersioningRail");
    versioning_collapsed_nav_->setAttribute(Qt::WA_StyledBackground, true);
    versioning_collapsed_nav_->setStyleSheet(version_rail_style);
    versioning_collapsed_nav_->setFixedWidth(kVersioningCollapsedWidth);
    auto* versioning_collapsed_layout = new QVBoxLayout(versioning_collapsed_nav_);
    versioning_collapsed_layout->setContentsMargins(7, 12, 7, 12);
    versioning_collapsed_layout->setSpacing(8);
    versioning_collapsed_layout->setAlignment(Qt::AlignTop);

    const auto configure_versioning_rail_button = [](QPushButton* button) {
      if (!button) {
        return;
      }
      button->setCursor(Qt::PointingHandCursor);
      button->setAutoDefault(false);
      button->setDefault(false);
      button->setFocusPolicy(Qt::NoFocus);
      button->setProperty("versioningActive", false);
      button->setFixedSize(kVersioningRailButtonSize, kVersioningRailButtonSize);
      button->setIconSize(kVersioningRailIconSize);
    };

    versioning_history_btn_ = new QPushButton(versioning_collapsed_nav_);
    configure_versioning_rail_button(versioning_history_btn_);
    versioning_collapsed_layout->addWidget(versioning_history_btn_, 0,
                                           Qt::AlignTop | Qt::AlignHCenter);

    versioning_versions_btn_ = new QPushButton(versioning_collapsed_nav_);
    configure_versioning_rail_button(versioning_versions_btn_);
    versioning_collapsed_layout->addWidget(versioning_versions_btn_, 0,
                                           Qt::AlignTop | Qt::AlignHCenter);

    versioning_collapsed_layout->addSpacing(6);
    auto* nav_divider = new QFrame(versioning_collapsed_nav_);
    nav_divider->setFrameShape(QFrame::HLine);
    nav_divider->setFixedWidth(kVersioningRailButtonSize - 10);
    nav_divider->setStyleSheet(QStringLiteral("color: %1; background: %1; border: none;")
                                   .arg(QColor(theme.dividerColor().red(),
                                               theme.dividerColor().green(),
                                               theme.dividerColor().blue(), 96)
                                            .name(QColor::HexArgb)));
    versioning_collapsed_layout->addWidget(nav_divider, 0, Qt::AlignHCenter);
    versioning_collapsed_layout->addStretch();

    QObject::connect(versioning_history_btn_, &QPushButton::clicked, this, [this]() {
      const bool same_page_open =
          !versioning_collapsed_ && versioning_active_page_ == VersioningFlyoutPage::History;
      versioning_active_page_ = VersioningFlyoutPage::History;
      if (same_page_open) {
        SetVersioningCollapsed(true, /*animate=*/false);
      } else {
        SetVersioningCollapsed(false, /*animate=*/false);
      }
    });
    QObject::connect(versioning_versions_btn_, &QPushButton::clicked, this, [this]() {
      const bool same_page_open =
          !versioning_collapsed_ && versioning_active_page_ == VersioningFlyoutPage::Versions;
      versioning_active_page_ = VersioningFlyoutPage::Versions;
      if (same_page_open) {
        SetVersioningCollapsed(true, /*animate=*/false);
      } else {
        SetVersioningCollapsed(false, /*animate=*/false);
      }
    });

    versioning_host_layout->addWidget(versioning_collapsed_nav_, 0);

    // ── Floating versioning flyout (not in splitter) ───────────────────────
    // This QWidget is a child of the dialog but not in any layout; it is
    // manually positioned via RepositionVersioningFlyout() so it overlays
    // the viewer area when open.
    versioning_flyout_ = new QWidget(this);
    versioning_flyout_->setObjectName("EditorVersioningFlyout");
    versioning_flyout_->setAttribute(Qt::WA_StyledBackground, false);
    versioning_flyout_->setAttribute(Qt::WA_TranslucentBackground, true);
    versioning_flyout_->hide();
    versioning_flyout_->installEventFilter(this);

    auto* flyout_layout = new QVBoxLayout(versioning_flyout_);
    flyout_layout->setContentsMargins(0, 0, 0, 0);
    flyout_layout->setSpacing(0);

    shared_versioning_root_ = new QWidget(versioning_flyout_);
    shared_versioning_root_->setObjectName("EditorVersioningPanel");
    shared_versioning_root_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    shared_versioning_root_->setAttribute(Qt::WA_StyledBackground, true);
    shared_versioning_root_->setStyleSheet(version_panel_style);
    flyout_layout->addWidget(shared_versioning_root_, 1);

    versioning_panel_opacity_effect_ = new QGraphicsOpacityEffect(shared_versioning_root_);
    shared_versioning_root_->setGraphicsEffect(versioning_panel_opacity_effect_);
    versioning_panel_opacity_effect_->setOpacity(0.0);

    auto* shared_versioning_outer_layout = new QVBoxLayout(shared_versioning_root_);
    shared_versioning_outer_layout->setContentsMargins(0, 0, 0, 0);
    shared_versioning_outer_layout->setSpacing(0);
    auto* versioning_content = new QWidget(shared_versioning_root_);
    versioning_content->setAttribute(Qt::WA_StyledBackground, false);
    versioning_content->setStyleSheet("background: transparent;");
    shared_versioning_layout_ = new QVBoxLayout(versioning_content);
    shared_versioning_layout_->setContentsMargins(20, 20, 20, 20);
    shared_versioning_layout_->setSpacing(14);
    shared_versioning_outer_layout->addWidget(versioning_content, 1);

    // ── Splitter: [rail (fixed)] | [viewer] | [controls] ──────────────────
    main_splitter->addWidget(versioning_panel_host_);
    main_splitter->addWidget(viewer_container_);
    main_splitter->addWidget(controls_panel);
    main_splitter->setStretchFactor(0, 0);
    main_splitter->setStretchFactor(1, 1);
    main_splitter->setStretchFactor(2, 0);

    const int right_default_width = std::clamp(
        static_cast<int>(std::lround(static_cast<double>(width()) * 0.25)),
        controls_panel->minimumWidth(), controls_panel->maximumWidth());
    main_splitter->setSizes({kVersioningCollapsedWidth,
                             std::max(400, width() - right_default_width - kVersioningCollapsedWidth),
                             right_default_width});
    root->addWidget(main_splitter, 1);
}

}  // namespace alcedo::ui
