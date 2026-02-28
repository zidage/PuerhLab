// ctor_viewer_and_panels.inc -- Constructor section for EditorDialog.
// This file is #included from the constructor body in dialog.cpp.
// It shares the constructor scope and has access to all local variables.

    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(0);

    auto* main_splitter = new QSplitter(Qt::Horizontal, this);
    main_splitter->setObjectName("EditorMainSplitter");
    main_splitter->setChildrenCollapsible(false);
    main_splitter->setHandleWidth(8);
    main_splitter->setStyleSheet(
        "QSplitter#EditorMainSplitter::handle {"
        "  background: #2A2A2A;"
        "}"
        "QSplitter#EditorMainSplitter::handle:hover {"
        "  background: #FCC704;"
        "}");

    viewer_ = new QtEditViewer(this);
    viewer_->setMinimumSize(560, 360);
    viewer_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    viewer_->setStyleSheet(
        "QOpenGLWidget {"
        "  background: #121212;"
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
    viewer_zoom_label_->setStyleSheet(
        "QLabel {"
        "  color: #E6E6E6;"
        "  background: rgba(18, 18, 18, 180);"
        "  border: 1px solid rgba(42, 42, 42, 220);"
        "  border-radius: 8px;"
        "  padding: 4px 8px;"
        "  font-size: 11px;"
        "  font-weight: 600;"
        "}");
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
    controls_panel->setStyleSheet(
        "#EditorControlsPanel {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorControlsPanel QFrame#EditorSection {"
        "  background: #121212;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionTitle {"
        "  color: #E6E6E6;"
        "  font-size: 13px;"
        "  font-weight: 620;"
        "}"
        "#EditorControlsPanel QLabel#EditorSectionSub {"
        "  color: #A3A3A3;"
        "  font-size: 11px;"
        "}");
    auto* controls_panel_layout = new QVBoxLayout(controls_panel);
    controls_panel_layout->setContentsMargins(16, 16, 16, 16);
    controls_panel_layout->setSpacing(12);

    const QString scroll_style =
        "QScrollArea { background: transparent; border: none; }"
        "QScrollBar:vertical {"
        "  background: #121212;"
        "  width: 10px;"
        "  margin: 2px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #FCC704;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; "
        "}";

    tone_controls_scroll_ = new QScrollArea(controls_panel);
    tone_controls_scroll_->setFrameShape(QFrame::NoFrame);
    tone_controls_scroll_->setWidgetResizable(true);
    tone_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    tone_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    tone_controls_scroll_->setStyleSheet(scroll_style);

    tone_controls_        = new QWidget(tone_controls_scroll_);
    controls_             = tone_controls_;
    auto* controls_layout = new QVBoxLayout(tone_controls_);
    controls_layout->setContentsMargins(0, 0, 0, 0);
    controls_layout->setSpacing(12);
    tone_controls_scroll_->setWidget(tone_controls_);
    controls_scroll_ = tone_controls_scroll_;

    geometry_controls_scroll_ = new QScrollArea(controls_panel);
    geometry_controls_scroll_->setFrameShape(QFrame::NoFrame);
    geometry_controls_scroll_->setWidgetResizable(true);
    geometry_controls_scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    geometry_controls_scroll_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    geometry_controls_scroll_->setStyleSheet(scroll_style);

    geometry_controls_ = new QWidget(geometry_controls_scroll_);
    auto* geometry_controls_layout = new QVBoxLayout(geometry_controls_);
    geometry_controls_layout->setContentsMargins(0, 0, 0, 0);
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
    raw_controls_layout->setContentsMargins(0, 0, 0, 0);
    raw_controls_layout->setSpacing(12);
    raw_controls_layout->addStretch();
    raw_controls_scroll_->setWidget(raw_controls_);

    control_panels_stack_ = new QStackedWidget(controls_panel);
    control_panels_stack_->addWidget(tone_controls_scroll_);
    control_panels_stack_->addWidget(geometry_controls_scroll_);
    control_panels_stack_->addWidget(raw_controls_scroll_);
    control_panels_stack_->setCurrentIndex(0);

    auto* panel_switch_row = new QWidget(controls_panel);
    auto* panel_switch_layout = new QHBoxLayout(panel_switch_row);
    panel_switch_layout->setContentsMargins(0, 0, 0, 0);
    panel_switch_layout->setSpacing(8);

    tone_panel_btn_     = new QPushButton("Tone", panel_switch_row);
    geometry_panel_btn_ = new QPushButton("Geometry", panel_switch_row);
    raw_panel_btn_      = new QPushButton("RAW Decode", panel_switch_row);
    tone_panel_btn_->setCheckable(true);
    geometry_panel_btn_->setCheckable(true);
    raw_panel_btn_->setCheckable(true);
    tone_panel_btn_->setCursor(Qt::PointingHandCursor);
    geometry_panel_btn_->setCursor(Qt::PointingHandCursor);
    raw_panel_btn_->setCursor(Qt::PointingHandCursor);
    tone_panel_btn_->setFixedHeight(30);
    geometry_panel_btn_->setFixedHeight(30);
    raw_panel_btn_->setFixedHeight(30);

    panel_switch_layout->addWidget(tone_panel_btn_, 1);
    panel_switch_layout->addWidget(geometry_panel_btn_, 1);
    panel_switch_layout->addWidget(raw_panel_btn_, 1);

    QObject::connect(tone_panel_btn_, &QPushButton::clicked, this,
                     [this]() { SetActiveControlPanel(ControlPanelKind::Tone); });
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
    shared_versioning_root->setStyleSheet(
        "#EditorVersioningPanel {"
        "  background: #1A1A1A;"
        "  border: none;"
        "  border-radius: 14px;"
        "}"
        "#EditorVersioningPanel QFrame#EditorSection {"
        "  background: #121212;"
        "  border: none;"
        "  border-radius: 12px;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionTitle {"
        "  color: #E6E6E6;"
        "  font-size: 13px;"
        "  font-weight: 620;"
        "}"
        "#EditorVersioningPanel QLabel#EditorSectionSub {"
        "  color: #A3A3A3;"
        "  font-size: 11px;"
        "}");
    auto* shared_versioning_outer_layout = new QVBoxLayout(shared_versioning_root);
    shared_versioning_outer_layout->setContentsMargins(0, 0, 0, 0);
    shared_versioning_outer_layout->setSpacing(0);

    auto* versioning_scroll = new QScrollArea(shared_versioning_root);
    versioning_scroll->setFrameShape(QFrame::NoFrame);
    versioning_scroll->setWidgetResizable(true);
    versioning_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    versioning_scroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    versioning_scroll->setStyleSheet(
        "QScrollArea { background: transparent; border: none; }"
        "QScrollBar:vertical {"
        "  background: #121212;"
        "  width: 10px;"
        "  margin: 2px;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::handle:vertical {"
        "  background: #FCC704;"
        "  border-radius: 5px;"
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }");

    auto* versioning_content = new QWidget(versioning_scroll);
    versioning_content->setStyleSheet("background: transparent;");
    auto* shared_versioning_layout = new QVBoxLayout(versioning_content);
    shared_versioning_layout->setContentsMargins(16, 16, 16, 16);
    shared_versioning_layout->setSpacing(8);
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

    {
      auto* histogram_frame = new QFrame(controls_panel);
      histogram_frame->setObjectName("EditorSection");
      auto* histogram_layout = new QVBoxLayout(histogram_frame);
      histogram_layout->setContentsMargins(12, 10, 12, 10);
      histogram_layout->setSpacing(6);

      auto* histogram_title = new QLabel("Histogram", histogram_frame);
      histogram_title->setObjectName("EditorSectionTitle");

      histogram_widget_       = new HistogramWidget(viewer_, histogram_frame);
      histogram_ruler_widget_ = new HistogramRulerWidget(
          viewer_ ? viewer_->GetHistogramBinCount() : 256, histogram_frame);
      histogram_layout->addWidget(histogram_title, 0);
      histogram_layout->addWidget(histogram_widget_, 0);
      histogram_layout->addWidget(histogram_ruler_widget_, 0);
      controls_panel_layout->addWidget(histogram_frame, 0);

      if (viewer_ && histogram_widget_) {
        viewer_->SetHistogramUpdateIntervalMs(histogram::kDefaultUpdateIntervalMs);
        viewer_->SetHistogramFrameExpected(false);
        QObject::connect(viewer_, &QtEditViewer::HistogramDataUpdated, histogram_widget_,
                         QOverload<>::of(&QWidget::update), Qt::QueuedConnection);
      }
    }

    controls_panel_layout->addWidget(panel_switch_row, 0);
    controls_panel_layout->addWidget(control_panels_stack_, 1);

