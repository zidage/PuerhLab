//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/editor_dialog/dialog_internal.hpp"
#include "ui/puerhlab_main/editor_dialog/widgets/editor_viewer_pane.hpp"

namespace puerhlab::ui {

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
        "  background: %2;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionTitle {"
        "  color: %3;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionSub {"
        "  color: %4;"
        "}")
                                           .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                                theme.bgBaseColor().name(QColor::HexArgb),
                                                theme.textColor().name(QColor::HexRgb),
                                                theme.textMutedColor().name(QColor::HexRgb));
    const QColor  rail_accent = theme.accentSecondaryColor();
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
        "  border: 1px solid %2;"
        "  background: %3;"
        "  padding: 0px;"
        "}"
        "#EditorVersioningRail QPushButton:hover {"
        "  background: %4;"
        "  border-color: %5;"
        "}"
        "#EditorVersioningRail QPushButton:pressed {"
        "  background: %6;"
        "}"
        "#EditorVersioningRail QPushButton:checked {"
        "  background: %7;"
        "  border-color: %8;"
        "}"
        "#EditorVersioningRail QPushButton:checked:hover {"
        "  background: %9;"
        "}")
                                          .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 168)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 46)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 82)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 220)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 112)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 230)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 255)
                                                   .name(QColor::HexArgb),
                                               QColor(rail_accent.red(), rail_accent.green(),
                                                      rail_accent.blue(), 255)
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

    // Viewer container allows a small overlay spinner during rendering.
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

    viewer_zoom_label_ = new QLabel(viewer_container_);
    viewer_zoom_label_->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    AppTheme::MarkFontRole(viewer_zoom_label_, AppTheme::FontRole::DataOverlay);
    viewer_zoom_label_->setStyleSheet(QStringLiteral(
                                          "QLabel {"
                                          "  color: %1;"
                                          "  background: %2;"
                                          "  border: 1px solid %3;"
                                          "  border-radius: 10px;"
                                          "  padding: 4px 8px;"
                                          "}")
                                          .arg(theme.textColor().name(QColor::HexRgb),
                                               QColor(21, 26, 32, 184).name(QColor::HexArgb),
                                               theme.glassStrokeColor().name(QColor::HexArgb)));
    viewer_grid->addWidget(viewer_zoom_label_, 0, 0, Qt::AlignLeft | Qt::AlignTop);

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

    versioning_panel_host_ = new VersioningPanelWidget(this);
    versioning_panel_host_->setMinimumWidth(kVersioningCollapsedWidth);
    versioning_panel_host_->setMaximumWidth(kVersioningCollapsedWidth + kVersioningExpandedMaxWidth);
    versioning_panel_host_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    auto* versioning_host_layout = new QHBoxLayout(versioning_panel_host_);
    versioning_host_layout->setContentsMargins(0, 0, 0, 0);
    versioning_host_layout->setSpacing(0);

    versioning_panel_content_ = new QWidget(versioning_panel_host_);
    versioning_panel_content_->setMinimumWidth(0);
    versioning_panel_content_->setMaximumWidth(0);
    versioning_panel_content_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    auto* versioning_panel_content_layout = new QVBoxLayout(versioning_panel_content_);
    versioning_panel_content_layout->setContentsMargins(0, 0, 0, 0);
    versioning_panel_content_layout->setSpacing(0);

    shared_versioning_root_ = new QWidget(versioning_panel_content_);
    shared_versioning_root_->setObjectName("EditorVersioningPanel");
    shared_versioning_root_->setMinimumWidth(kVersioningExpandedMinWidth);
    shared_versioning_root_->setMaximumWidth(kVersioningExpandedMaxWidth);
    shared_versioning_root_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    shared_versioning_root_->setAttribute(Qt::WA_StyledBackground, true);
    shared_versioning_root_->setStyleSheet(version_panel_style);
    versioning_panel_content_layout->addWidget(shared_versioning_root_, 1);

    versioning_panel_opacity_effect_ = new QGraphicsOpacityEffect(shared_versioning_root_);
    shared_versioning_root_->setGraphicsEffect(versioning_panel_opacity_effect_);
    versioning_panel_opacity_effect_->setOpacity(0.0);

    auto* shared_versioning_outer_layout = new QVBoxLayout(shared_versioning_root_);
    shared_versioning_outer_layout->setContentsMargins(0, 0, 0, 0);
    shared_versioning_outer_layout->setSpacing(0);

    auto* versioning_scroll = new QScrollArea(shared_versioning_root_);
    versioning_scroll->setFrameShape(QFrame::NoFrame);
    versioning_scroll->setWidgetResizable(true);
    versioning_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    versioning_scroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    versioning_scroll->setStyleSheet(AppTheme::EditorScrollAreaStyle());

    auto* versioning_content = new QWidget(versioning_scroll);
    versioning_content->setStyleSheet("background: transparent;");
    shared_versioning_layout_ = new QVBoxLayout(versioning_content);
    shared_versioning_layout_->setContentsMargins(20, 20, 20, 20);
    shared_versioning_layout_->setSpacing(14);
    versioning_scroll->setWidget(versioning_content);
    shared_versioning_outer_layout->addWidget(versioning_scroll, 1);

    versioning_collapsed_nav_ = new QWidget(versioning_panel_host_);
    versioning_collapsed_nav_->setObjectName("EditorVersioningRail");
    versioning_collapsed_nav_->setAttribute(Qt::WA_StyledBackground, true);
    versioning_collapsed_nav_->setStyleSheet(version_rail_style);
    versioning_collapsed_nav_->setFixedWidth(kVersioningCollapsedWidth);
    auto* versioning_collapsed_layout = new QVBoxLayout(versioning_collapsed_nav_);
    versioning_collapsed_layout->setContentsMargins(7, 12, 7, 12);
    versioning_collapsed_layout->setSpacing(8);
    versioning_collapsed_layout->setAlignment(Qt::AlignTop);

    versioning_nav_btn_ = new QPushButton(versioning_collapsed_nav_);
    versioning_nav_btn_->setCursor(Qt::PointingHandCursor);
    versioning_nav_btn_->setCheckable(true);
    versioning_nav_btn_->setAutoDefault(false);
    versioning_nav_btn_->setDefault(false);
    versioning_nav_btn_->setAttribute(Qt::WA_Hover, true);
    versioning_nav_btn_->setFixedSize(kVersioningRailButtonSize, kVersioningRailButtonSize);
    versioning_nav_btn_->setIconSize(kVersioningRailIconSize);
    versioning_nav_btn_->installEventFilter(this);
    versioning_collapsed_layout->addWidget(versioning_nav_btn_, 0, Qt::AlignTop | Qt::AlignHCenter);
    versioning_collapsed_layout->addStretch();

    QObject::connect(versioning_nav_btn_, &QPushButton::clicked, this,
                     [this]() { SetVersioningCollapsed(versioning_panel_progress_ > 0.5); });

    versioning_collapsed_gap_base_width_ = std::max(0, kEditorOuterMargin - main_splitter->handleWidth());
    versioning_collapsed_gap_ = new QWidget(versioning_panel_host_);
    versioning_collapsed_gap_->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    versioning_collapsed_gap_->setFixedWidth(versioning_collapsed_gap_base_width_);
    versioning_collapsed_gap_->setVisible(versioning_collapsed_gap_base_width_ > 0);

    versioning_host_layout->addWidget(versioning_collapsed_nav_, 0);
    versioning_host_layout->addWidget(versioning_panel_content_, 0);
    versioning_host_layout->addWidget(versioning_collapsed_gap_, 0);

    main_splitter->addWidget(versioning_panel_host_);
    main_splitter->addWidget(viewer_container_);
    main_splitter->addWidget(controls_panel);
    main_splitter->setStretchFactor(0, 0);
    main_splitter->setStretchFactor(1, 1);
    main_splitter->setStretchFactor(2, 0);
    const int right_default_width = std::clamp(
        static_cast<int>(std::lround(static_cast<double>(width()) * 0.25)),
        controls_panel->minimumWidth(), controls_panel->maximumWidth());
    const int left_default_width = 320;
    versioning_expanded_width_ =
        std::clamp(left_default_width - kVersioningCollapsedWidth, kVersioningExpandedMinWidth,
                   kVersioningExpandedMaxWidth);
    const int collapsed_left_width = kVersioningCollapsedWidth + versioning_collapsed_gap_base_width_;
    main_splitter->setSizes({collapsed_left_width,
                             std::max(400, width() - right_default_width - collapsed_left_width),
                             right_default_width});
    root->addWidget(main_splitter, 1);

}

}  // namespace puerhlab::ui
