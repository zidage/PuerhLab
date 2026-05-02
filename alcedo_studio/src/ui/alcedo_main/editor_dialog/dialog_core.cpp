#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {
namespace {

auto LoadScopeExifDisplayMetaData(const std::shared_ptr<ImagePoolService>& image_pool,
                                  image_id_t image_id) -> ExifDisplayMetaData {
  if (!image_pool || image_id == 0) {
    return {};
  }

  try {
    return image_pool->Read<ExifDisplayMetaData>(
        image_id, [](const std::shared_ptr<Image>& image) -> ExifDisplayMetaData {
          if (!image) {
            return {};
          }
          if (image->has_exif_display_.load()) {
            return image->exif_display_;
          }
          if (image->has_exif_json_.load()) {
            ExifDisplayMetaData metadata;
            metadata.FromJson(image->exif_json_);
            return metadata;
          }
          return {};
        });
  } catch (...) {
    return {};
  }
}

}  // namespace

EditorDialog::EditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
               std::shared_ptr<PipelineGuard>          pipeline_guard,
               std::shared_ptr<EditHistoryMgmtService> history_service,
               std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
               image_id_t image_id, QWidget* parent)
      : QDialog(parent),
        image_pool_(std::move(image_pool)),
        pipeline_guard_(std::move(pipeline_guard)),
        history_service_(std::move(history_service)),
        history_guard_(std::move(history_guard)),
        element_id_(element_id),
        image_id_(image_id),
        scheduler_(RenderService::GetPreviewScheduler()) {
    if (!image_pool_ || !pipeline_guard_ || !pipeline_guard_->pipeline_ || !history_service_ ||
        !history_guard_ || !history_guard_->history_ || !scheduler_) {
      throw std::runtime_error("EditorDialog: missing services");
    }
    adjustment_session_ = std::make_unique<EditorAdjustmentSession>(
        EditorAdjustmentSession::Dependencies{
            .pipeline_guard  = pipeline_guard_,
            .working_version = &working_version_,
            .state           = &state_,
            .committed_state = &committed_state_,
        },
        EditorAdjustmentSession::Callbacks{
            .schedule_quality_preview =
                [this]() { ScheduleQualityPreviewRenderFromPipeline(); },
            .advance_preview_generation = [this]() { AdvancePreviewGeneration(); },
            .update_version_ui          = [this]() { UpdateVersionUi(); },
            .mark_full_frame_preview_after_geometry_commit =
                [this]() { frame_manager_.MarkNeedsFullFramePreviewAfterGeometryCommit(); },
        });

    setModal(true);
    setSizeGripEnabled(true);
    setWindowFlag(Qt::WindowMinMaxButtonsHint, true);
    setWindowFlag(Qt::MSWindowsFixedSizeDialogHint, false);
    setWindowTitle(ResolveEditorWindowTitle(image_pool_, image_id_, element_id_));
    setMinimumSize(1080, 680);
    resize(1500, 1000);
    AppTheme::ApplyFont(this, AppTheme::FontRole::UiBody);


    // --- EditorDialog UI construction. ---
    BuildViewerAndPanelShell();
    if (scope_panel_) {
      exif_display_ = LoadScopeExifDisplayMetaData(image_pool_, image_id_);
      scope_panel_->SetExifDisplayMetaData(exif_display_);
      UpdateViewerZoomLabel(viewer_ ? viewer_->GetViewZoom() : 1.0f);
    }
    BuildToneControlPanel();
    BuildDisplayTransformPanel();
    BuildGeometryRawPanels();
    BuildVersioningPanel();

    shortcut_registry_ = std::make_unique<ShortcutRegistry>(this);
    RegisterShortcuts();

    AppTheme::ApplyFontsRecursively(this);

    UpdateVersionUi();

    frame_manager_.SetViewer(viewer_);
    frame_manager_.SetScopePanel(scope_panel_);
    if (scope_panel_) {
      scope_panel_->SetNeedsRenderCallback(
          [this]() { RequestRender(/*use_viewport_region=*/true,
                                   /*bump_preview_generation=*/false); });
    }
    SetupPipeline();
    pipeline_initialized_ = true;
    if (viewer_) {
      viewer_->SetCropOverlayAspectLock(false, 1.0f);
      viewer_->SetCropOverlayRectNormalized(state_.crop_x_, state_.crop_y_, state_.crop_w_,
                                            state_.crop_h_);
      viewer_->SetCropOverlayRotationDegrees(state_.rotate_degrees_);
      viewer_->SetCropOverlayVisible(false);
      viewer_->SetCropToolEnabled(false);
    }
    if (viewer_) {
      QObject::connect(viewer_, &QtEditViewer::ViewInteractionSettled, this,
                       [this]() { MaybeScheduleDetailPreviewRenderFromViewport(); });
    }

    // Load a 4K quality base preview first; scheduler transitions back to fast-preview baseline.
    QTimer::singleShot(0, this, [this]() {
      AdvancePreviewGeneration();
      TriggerQualityPreviewRenderFromPipeline();
    });
  }

  void EditorDialog::RegisterShortcuts() {
    if (!shortcut_registry_) {
      return;
    }

    shortcut_registry_->Register({
        .id               = kShortcutUndoHistoryId,
        .description      = Tr("Undo last uncommitted transaction"),
        .default_sequence = QKeySequence(QKeySequence::Undo),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (!ShouldConsumeUndoShortcutLocally()) {
            UndoLastTransaction();
          }
        },
    });
    shortcut_registry_->Register({
        .id               = kShortcutResetGeometryId,
        .description      = Tr("Reset crop & rotation"),
        .default_sequence = QKeySequence(Qt::CTRL | Qt::Key_R),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() { ResetCropAndRotation(); },
    });
    shortcut_registry_->Register({
        .id               = kShortcutSelectPrevLutId,
        .description      = Tr("Select previous LUT"),
        .default_sequence = QKeySequence(Qt::Key_Up),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (ShouldConsumeLutNavigationShortcut()) {
            return;
          }
          if (lut_browser_widget_) {
            lut_browser_widget_->SelectRelativeEntry(-1);
          }
        },
    });
    shortcut_registry_->Register({
        .id               = kShortcutSelectNextLutId,
        .description      = Tr("Select next LUT"),
        .default_sequence = QKeySequence(Qt::Key_Down),
        .context          = Qt::WidgetWithChildrenShortcut,
        .on_trigger       = [this]() {
          if (ShouldConsumeLutNavigationShortcut()) {
            return;
          }
          if (lut_browser_widget_) {
            lut_browser_widget_->SelectRelativeEntry(1);
          }
        },
    });

    if (undo_tx_btn_) {
      undo_tx_btn_->setToolTip(shortcut_registry_->DecorateTooltip(
          Tr("Undo last uncommitted transaction"), kShortcutUndoHistoryId));
    }
    if (geometry_reset_btn_) {
      geometry_reset_btn_->setToolTip(
          shortcut_registry_->DecorateTooltip(Tr("Reset crop & rotation"),
                                             kShortcutResetGeometryId));
    }
  }

auto EditorDialog::ShouldConsumeUndoShortcutLocally() const -> bool {
    QWidget* const focus_widget = QApplication::focusWidget();
    if (!focus_widget || !isAncestorOf(focus_widget)) {
      return false;
    }

    return qobject_cast<QLineEdit*>(focus_widget) != nullptr ||
           qobject_cast<QTextEdit*>(focus_widget) != nullptr ||
           qobject_cast<QPlainTextEdit*>(focus_widget) != nullptr ||
           qobject_cast<QAbstractSpinBox*>(focus_widget) != nullptr;
  }

auto EditorDialog::ShouldConsumeLutNavigationShortcut() const -> bool {
    if (active_panel_ != ControlPanelKind::Look || !lut_browser_widget_) {
      return true;
    }
    if (QApplication::activePopupWidget()) {
      return true;
    }

    QWidget* const focus_widget = QApplication::focusWidget();
    if (!focus_widget || !isAncestorOf(focus_widget)) {
      return false;
    }

    if (qobject_cast<QComboBox*>(focus_widget) != nullptr ||
        qobject_cast<QAbstractSpinBox*>(focus_widget) != nullptr ||
        qobject_cast<QTextEdit*>(focus_widget) != nullptr ||
        qobject_cast<QPlainTextEdit*>(focus_widget) != nullptr) {
      return true;
    }
    if (auto* line_edit = qobject_cast<QLineEdit*>(focus_widget)) {
      return !lut_browser_widget_->isAncestorOf(line_edit);
    }
    return false;
  }

bool EditorDialog::eventFilter(QObject* obj, QEvent* event) {
    if (obj == versioning_flyout_ && event && event->type() == QEvent::Hide) {
      if (!versioning_collapsed_) {
        if (versioning_panel_anim_) {
          versioning_panel_anim_->stop();
        }
        versioning_panel_progress_ = 0.0;
        versioning_collapsed_      = true;
        if (versioning_panel_opacity_effect_) {
          versioning_panel_opacity_effect_->setOpacity(0.0);
        }
        RefreshVersioningCollapseUi();
      }
    }

    if (event && event->type() == QEvent::MouseButtonDblClick) {
      if (auto* slider = qobject_cast<QSlider*>(obj)) {
        const auto it = slider_reset_callbacks_.find(slider);
        if (it != slider_reset_callbacks_.end()) {
          if (!syncing_controls_ && it->second) {
            it->second();
          }
          return true;
        }
      }
      if (obj == curve_widget_ && curve_reset_callback_) {
        if (!syncing_controls_) {
          curve_reset_callback_();
        }
        return true;
      }
    }
    return QDialog::eventFilter(obj, event);
  }

void EditorDialog::changeEvent(QEvent* event) {
    if (event && event->type() == QEvent::LanguageChange) {
      RetranslateUi();
    }
    QDialog::changeEvent(event);
  }

void EditorDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    if (initial_splitter_sizes_applied_) {
      return;
    }
    initial_splitter_sizes_applied_ = true;
    QTimer::singleShot(50, this, [this]() { ApplyInitialSplitterSizes(); });
  }

void EditorDialog::ApplyInitialSplitterSizes() {
    if (!main_splitter_) {
      return;
    }

    QWidget* controls_panel = main_splitter_->widget(2);
    if (!controls_panel) {
      return;
    }

    const int handle_width = main_splitter_->handleWidth();
    const int available_width =
        std::max(0, main_splitter_->width() - handle_width * (main_splitter_->count() - 1));
    if (available_width <= 0) {
      return;
    }

    const int right_width = std::clamp(
        static_cast<int>(std::lround(static_cast<double>(available_width) * 0.25)),
        controls_panel->minimumWidth(), controls_panel->maximumWidth());
    const int center_width =
        std::max(400, available_width - kVersioningCollapsedWidth - right_width);

    main_splitter_->setSizes({kVersioningCollapsedWidth, center_width, right_width});
  }

void EditorDialog::resizeEvent(QResizeEvent* event) {
    QDialog::resizeEvent(event);
    if (versioning_flyout_ && versioning_flyout_->isVisible()) {
      RepositionVersioningFlyout();
    }
  }

void EditorDialog::RetranslateUi() {
    setWindowTitle(ResolveEditorWindowTitle(image_pool_, image_id_, element_id_));
    RetranslateMarkedObjects(this);

    if (tone_panel_btn_) {
      tone_panel_btn_->setText(Tr("Tone"));
    }
    if (look_panel_btn_) {
      look_panel_btn_->setText(Tr("Color"));
    }
    if (drt_panel_btn_) {
      drt_panel_btn_->setText(Tr("Display RT"));
      drt_panel_btn_->setToolTip(Tr("Display Rendering Transform"));
    }
    if (geometry_panel_btn_) {
      geometry_panel_btn_->setText(Tr("Geometry"));
    }
    if (raw_panel_btn_) {
      raw_panel_btn_->setText(Tr("RAW Decode"));
    }
    if (geometry_apply_btn_) {
      geometry_apply_btn_->setText(Tr("Apply Crop"));
    }
    if (geometry_reset_btn_) {
      geometry_reset_btn_->setText(Tr("Reset"));
      geometry_reset_btn_->setToolTip(Tr("Reset crop & rotation (Ctrl+R)"));
    }
    if (undo_tx_btn_) {
      undo_tx_btn_->setText(Tr("Undo Last"));
    }
    if (commit_version_btn_) {
      commit_version_btn_->setText(Tr("Commit All"));
    }
    if (new_working_btn_) {
      new_working_btn_->setText(Tr("New Working"));
    }
    if (color_temp_unsupported_label_) {
      color_temp_unsupported_label_->setText(
          Tr("Color temperature/tint is unavailable for this image."));
    }
    if (color_temp_mode_combo_) {
      const int current_value = color_temp_mode_combo_->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      color_temp_mode_combo_->clear();
      color_temp_mode_combo_->addItem(Tr("As Shot"), static_cast<int>(ColorTempMode::AS_SHOT));
      color_temp_mode_combo_->addItem(Tr("Custom"), static_cast<int>(ColorTempMode::CUSTOM));
      const int index = color_temp_mode_combo_->findData(current_value);
      color_temp_mode_combo_->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    }
    if (working_mode_combo_) {
      const int current_value = working_mode_combo_->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      working_mode_combo_->clear();
      working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
      working_mode_combo_->addItem(Tr("Incremental"),
                                   static_cast<int>(WorkingMode::Incremental));
      const int index = working_mode_combo_->findData(current_value);
      working_mode_combo_->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    }
    if (geometry_crop_aspect_preset_combo_) {
      const int current_value = geometry_crop_aspect_preset_combo_->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      geometry_crop_aspect_preset_combo_->clear();
      for (const auto& option : geometry::CropAspectPresetOptions()) {
        geometry_crop_aspect_preset_combo_->addItem(Tr(option.label_),
                                                    static_cast<int>(option.value_));
      }
      const int index = geometry_crop_aspect_preset_combo_->findData(current_value);
      geometry_crop_aspect_preset_combo_->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    }
    auto refresh_odt_combo = [this](QComboBox* combo, const auto& options) {
      if (!combo) {
        return;
      }
      const int current_value = combo->currentData().toInt();
      const bool prev_sync = syncing_controls_;
      syncing_controls_ = true;
      combo->clear();
      for (const auto& option : options) {
        combo->addItem(Tr(option.label_), static_cast<int>(option.value_));
      }
      const int index = combo->findData(current_value);
      combo->setCurrentIndex(std::max(0, index));
      syncing_controls_ = prev_sync;
    };
    refresh_odt_combo(odt_encoding_space_combo_, kDisplayEncodingSpaceOptions);
    refresh_odt_combo(odt_aces_limiting_space_combo_, kAcesLimitingSpaceOptions);
    refresh_odt_combo(odt_open_drt_look_preset_combo_, kOpenDrtLookPresetOptions);
    refresh_odt_combo(odt_open_drt_tonescale_preset_combo_, kOpenDrtTonescaleOptions);
    refresh_odt_combo(odt_open_drt_creative_white_combo_, kOpenDrtCreativeWhiteOptions);
    if (lut_browser_widget_) {
      lut_browser_widget_->RetranslateUi();
      RefreshLutBrowserUi();
    }

    RefreshLensComboFromState();
    RefreshOdtEncodingEotfComboFromState();
    RefreshHlsTargetUi();
    UpdateGeometryCropRectLabel();
    UpdateViewerZoomLabel(viewer_ ? viewer_->GetViewZoom() : 1.0f);
    RefreshVersioningCollapseUi();
    UpdateVersionUi();
  }
auto RunEditorDialog(std::shared_ptr<ImagePoolService>       image_pool,
                     std::shared_ptr<PipelineGuard>          pipeline_guard,
                     std::shared_ptr<EditHistoryMgmtService> history_service,
                     std::shared_ptr<EditHistoryGuard> history_guard, sl_element_id_t element_id,
                     image_id_t image_id, QWidget* parent) -> bool {
  EditorDialog dlg(std::move(image_pool), std::move(pipeline_guard), std::move(history_service),
                   std::move(history_guard), element_id, image_id, parent);
  dlg.showMaximized();
  dlg.exec();
  return true;
}

}  // namespace alcedo::ui
