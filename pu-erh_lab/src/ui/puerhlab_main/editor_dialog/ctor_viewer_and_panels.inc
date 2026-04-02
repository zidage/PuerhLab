// ctor_viewer_and_panels.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

    const auto& theme = AppTheme::Instance();
    const QString borderless_panel_style = QStringLiteral(
        "#EditorControlsPanel {"
        "  background: %1;"
        "  border: 1px solid %2;"
        "  border-radius: 14px;"
        "}"
        "#EditorControlsPanel QFrame#EditorSection {"
        "  background: %3;"
        "  border: 1px solid %4;"
        "  border-radius: 12px;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionTitle {"
        "  color: %5;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionSub {"
        "  color: %6;"
        "}")
                                             .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                                  theme.glassStrokeColor().name(QColor::HexArgb),
                                                  theme.bgBaseColor().name(QColor::HexArgb),
                                                  theme.dividerColor().name(QColor::HexArgb),
                                                  theme.textColor().name(QColor::HexRgb),
                                                  theme.textMutedColor().name(QColor::HexRgb));
    const QString version_panel_style = QStringLiteral(
        "#EditorVersioningPanel {"
        "  background: %1;"
        "  border: 1px solid %2;"
        "  border-radius: 14px;"
        "}"
        "#EditorVersioningPanel QFrame#EditorSection {"
        "  background: %3;"
        "  border: 1px solid %4;"
        "  border-radius: 12px;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionTitle {"
        "  color: %5;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionSub {"
        "  color: %6;"
        "}")
                                           .arg(theme.glassPanelColor().name(QColor::HexArgb),
                                                theme.glassStrokeColor().name(QColor::HexArgb),
                                                theme.bgBaseColor().name(QColor::HexArgb),
                                                theme.dividerColor().name(QColor::HexArgb),
                                                theme.textColor().name(QColor::HexRgb),
                                                theme.textMutedColor().name(QColor::HexRgb));

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(0);

    auto* main_splitter = new QSplitter(Qt::Horizontal, this);
    main_splitter->setObjectName("EditorMainSplitter");
    main_splitter->setChildrenCollapsible(false);
    main_splitter->setHandleWidth(8);
    main_splitter->setStyleSheet(
        "QSplitter#EditorMainSplitter::handle {"
        "  background: rgba(233, 227, 216, 0.08);"
        "}"
        "QSplitter#EditorMainSplitter::handle:hover {"
        "  background: rgba(185, 138, 74, 0.7);"
        "}");

    viewer_ = new QtEditViewer(this);
    viewer_->setMinimumSize(560, 360);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QWidget {"
        "  background: #151A20;"
        "  border: none;"
        "  border-radius: 12px;"
        "}");

    // Viewer container allows a small overlay spinner during rendering.
    viewer_container_ = new QWidget(this);
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

    auto* controls_panel = new QWidget(this);
    controls_panel->setMinimumWidth(320);
    controls_panel->setMaximumWidth(900);
    controls_panel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    controls_panel->setObjectName("EditorControlsPanel");
    controls_panel->setAttribute(Qt::WA_StyledBackground, true);
    controls_panel->setStyleSheet(borderless_panel_style);
    auto* controls_panel_layout = new QVBoxLayout(controls_panel);
    controls_panel_layout->setContentsMargins(18, 18, 18, 18);
    controls_panel_layout->setSpacing(12);

    const QString scroll_style = AppTheme::EditorScrollAreaStyle();

    tone_controls_scroll_ = new QScrollArea(controls_panel);
    tone_controls_scroll_->setFrameShape(QFrame::NoFrame);
    tone_controls_scroll_->setWidgetResizable(true);
    tone_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tone_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    tone_controls_scroll_->setStyleSheet(scroll_style);

    tone_controls_        = new QWidget(tone_controls_scroll_);
    controls_             = tone_controls_;
    auto* controls_layout = new QVBoxLayout(tone_controls_);
    controls_layout->setContentsMargins(14, 14, 14, 14);
    controls_layout->setSpacing(12);
    tone_controls_scroll_->setWidget(tone_controls_);
    controls_scroll_ = tone_controls_scroll_;

    drt_controls_scroll_ = new QScrollArea(controls_panel);
    drt_controls_scroll_->setFrameShape(QFrame::NoFrame);
    drt_controls_scroll_->setWidgetResizable(true);
    drt_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    drt_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    drt_controls_scroll_->setStyleSheet(scroll_style);

    drt_controls_ = new QWidget(drt_controls_scroll_);
    auto* drt_controls_layout = new QVBoxLayout(drt_controls_);
    drt_controls_layout->setContentsMargins(14, 14, 14, 14);
    drt_controls_layout->setSpacing(12);
    drt_controls_layout->addStretch();
    drt_controls_scroll_->setWidget(drt_controls_);

    look_controls_scroll_ = new QScrollArea(controls_panel);
    look_controls_scroll_->setFrameShape(QFrame::NoFrame);
    look_controls_scroll_->setWidgetResizable(true);
    look_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    look_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    look_controls_scroll_->setStyleSheet(scroll_style);

    look_controls_ = new QWidget(look_controls_scroll_);
    auto* look_controls_layout = new QVBoxLayout(look_controls_);
    look_controls_layout->setContentsMargins(14, 14, 14, 14);
    look_controls_layout->setSpacing(12);

    lut_browser_widget_ = new LutBrowserWidget(look_controls_);
    look_controls_layout->addWidget(lut_browser_widget_, 1);
    look_controls_scroll_->setWidget(look_controls_);

    geometry_controls_scroll_ = new QScrollArea(controls_panel);
    geometry_controls_scroll_->setFrameShape(QFrame::NoFrame);
    geometry_controls_scroll_->setWidgetResizable(true);
    geometry_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    geometry_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    geometry_controls_scroll_->setStyleSheet(scroll_style);

    geometry_controls_ = new QWidget(geometry_controls_scroll_);
    auto* geometry_controls_layout = new QVBoxLayout(geometry_controls_);
    geometry_controls_layout->setContentsMargins(14, 14, 14, 14);
    geometry_controls_layout->setSpacing(12);
    geometry_controls_layout->addStretch();
    geometry_controls_scroll_->setWidget(geometry_controls_);

    raw_controls_scroll_ = new QScrollArea(controls_panel);
    raw_controls_scroll_->setFrameShape(QFrame::NoFrame);
    raw_controls_scroll_->setWidgetResizable(true);
    raw_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    raw_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    raw_controls_scroll_->setStyleSheet(scroll_style);

    raw_controls_ = new QWidget(raw_controls_scroll_);
    auto* raw_controls_layout = new QVBoxLayout(raw_controls_);
    raw_controls_layout->setContentsMargins(14, 14, 14, 14);
    raw_controls_layout->setSpacing(12);
    raw_controls_layout->addStretch();
    raw_controls_scroll_->setWidget(raw_controls_);

    control_panels_stack_ = new QStackedWidget(controls_panel);
    control_panels_stack_->addWidget(tone_controls_scroll_);
    control_panels_stack_->addWidget(look_controls_scroll_);
    control_panels_stack_->addWidget(drt_controls_scroll_);
    control_panels_stack_->addWidget(geometry_controls_scroll_);
    control_panels_stack_->addWidget(raw_controls_scroll_);
    control_panels_stack_->setCurrentIndex(0);

    auto* panel_switch_row = new QWidget(controls_panel);
    panel_switch_row->setObjectName("EditorPanelSwitchRow");
    panel_switch_row->setAttribute(Qt::WA_StyledBackground, true);
    panel_switch_row->setStyleSheet(
        "#EditorPanelSwitchRow {"
        "  background: rgba(21, 26, 32, 0.84);"
        "  border: 1px solid rgba(233, 227, 216, 0.08);"
        "  border-radius: 10px;"
        "}");
    auto* panel_switch_layout = new QHBoxLayout(panel_switch_row);
    panel_switch_layout->setContentsMargins(0, 0, 0, 0);
    panel_switch_layout->setSpacing(0);

    tone_panel_btn_     = new QPushButton(panel_switch_row);
    look_panel_btn_     = new QPushButton(panel_switch_row);
    drt_panel_btn_      = new QPushButton(panel_switch_row);
    geometry_panel_btn_ = new QPushButton(panel_switch_row);
    raw_panel_btn_      = new QPushButton(panel_switch_row);

    ConfigurePanelToggleButton(tone_panel_btn_, Tr("Tone"),
                               QStringLiteral(":/panel_icons/adjustments.svg"));
    ConfigurePanelToggleButton(look_panel_btn_, Tr("Look"),
                               QStringLiteral(":/panel_icons/palette.svg"));
    ConfigurePanelToggleButton(drt_panel_btn_, Tr("Display Rendering Transform"),
                               QStringLiteral(":/panel_icons/color-filter.svg"));
    ConfigurePanelToggleButton(geometry_panel_btn_, Tr("Geometry"),
                               QStringLiteral(":/panel_icons/crop.svg"));
    ConfigurePanelToggleButton(raw_panel_btn_, Tr("RAW Decode"),
                               QStringLiteral(":/panel_icons/aperture.svg"));

    panel_switch_layout->addWidget(tone_panel_btn_, 1);
    panel_switch_layout->addWidget(look_panel_btn_, 1);
    panel_switch_layout->addWidget(drt_panel_btn_, 1);
    panel_switch_layout->addWidget(geometry_panel_btn_, 1);
    panel_switch_layout->addWidget(raw_panel_btn_, 1);

    QObject::connect(tone_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Tone); });
    QObject::connect(look_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Look); });
    QObject::connect(drt_panel_btn_, &QPushButton::clicked, this,
                     [this]() {
                       SetActiveControlPanel(ControlPanelKind::DisplayRenderingTransform);
                     });
    QObject::connect(geometry_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Geometry); });
    QObject::connect(raw_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::RawDecode); });
    RefreshPanelSwitchUi();

    auto* shared_versioning_root = new QWidget(this);
    shared_versioning_root->setObjectName("EditorVersioningPanel");
    shared_versioning_root->setMinimumWidth(220);
    shared_versioning_root->setMaximumWidth(600);
    shared_versioning_root->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    shared_versioning_root->setAttribute(Qt::WA_StyledBackground, true);
    shared_versioning_root->setStyleSheet(version_panel_style);
    auto* shared_versioning_outer_layout = new QVBoxLayout(shared_versioning_root);
    shared_versioning_outer_layout->setContentsMargins(0, 0, 0, 0);
    shared_versioning_outer_layout->setSpacing(0);

    auto* versioning_scroll = new QScrollArea(shared_versioning_root);
    versioning_scroll->setFrameShape(QFrame::NoFrame);
    versioning_scroll->setWidgetResizable(true);
    versioning_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    versioning_scroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    versioning_scroll->setStyleSheet(AppTheme::EditorScrollAreaStyle());

    auto* versioning_content = new QWidget(versioning_scroll);
    versioning_content->setStyleSheet("background: transparent;");
    auto* shared_versioning_layout = new QVBoxLayout(versioning_content);
    shared_versioning_layout->setContentsMargins(18, 18, 18, 18);
    shared_versioning_layout->setSpacing(10);
    versioning_scroll->setWidget(versioning_content);
    shared_versioning_outer_layout->addWidget(versioning_scroll, 1);

    main_splitter->addWidget(shared_versioning_root);
    main_splitter->addWidget(viewer_container_);
    main_splitter->addWidget(controls_panel);
    main_splitter->setStretchFactor(0, 0);
    main_splitter->setStretchFactor(1, 1);
    main_splitter->setStretchFactor(2, 0);
    const int right_default_width =
        std::clamp(width() / 3, controls_panel->minimumWidth(), controls_panel->maximumWidth());
    const int left_default_width = 280;
    main_splitter->setSizes({left_default_width,
                             std::max(400, width() - right_default_width - left_default_width),
                             right_default_width});
    root->addWidget(main_splitter, 1);

    auto* scope_frame = new QFrame(controls_panel);
    scope_frame->setObjectName("EditorSection");
    auto* scope_layout = new QVBoxLayout(scope_frame);
    scope_layout->setContentsMargins(0, 0, 0, 0);
    scope_layout->setSpacing(0);
    scope_panel_ = new ScopePanel(scope_frame);
    scope_layout->addWidget(scope_panel_, 1);
    controls_panel_layout->addWidget(scope_frame, 0);

    controls_panel_layout->addWidget(panel_switch_row, 0);
    controls_panel_layout->addWidget(control_panels_stack_, 1);
