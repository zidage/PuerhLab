#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/display_transform_pipeline_adapter.hpp"
#include "ui/alcedo_main/editor_dialog/pipeline/raw_pipeline_adapter.hpp"

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
                           std::shared_ptr<EditHistoryGuard>       history_guard,
                           sl_element_id_t element_id, image_id_t image_id, QWidget* parent)
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
  history_coordinator_ = std::make_unique<EditorHistoryCoordinator>(
      EditorHistoryCoordinator::Dependencies{
          .history_service = history_service_,
          .history_guard   = history_guard_,
          .pipeline_guard  = pipeline_guard_,
          .element_id      = element_id_,
          .message_parent  = this,
      },
      EditorHistoryCoordinator::Callbacks{
          .reload_ui_state_from_pipeline =
              [this](bool reset_to_defaults_if_missing) {
                const bool loaded = LoadStateFromPipelineIfPresent();
                if (!loaded && !reset_to_defaults_if_missing) {
                  return false;
                }
                if (!loaded) {
                  state_ = AdjustmentState{};
                  SanitizeOdtStateForUi(state_.odt_);
                  UpdateAllCdlWheelDerivedColors(state_);
                  last_submitted_color_temp_request_.reset();
                } else {
                  last_submitted_color_temp_request_ = BuildColorTempRequest(state_);
                }
                committed_state_ = state_;
                SyncControlsFromState();
                AdvancePreviewGeneration();
                TriggerQualityPreviewRenderFromPipeline();
                return true;
              },
          .after_pipeline_params_imported =
              [this]() {
                frame_manager_.AttachExecutionStages(pipeline_guard_->pipeline_);
                last_applied_lut_path_.clear();
              },
          .is_plain_working_mode = [this]() { return CurrentWorkingMode() == WorkingMode::Plain; },
          .refresh_version_log_selection_styles = [this]() { RefreshVersionLogSelectionStyles(); },
      });
  render_coordinator_ = std::make_unique<EditorRenderCoordinator>(
      EditorRenderCoordinator::Dependencies{
          .timer_parent   = this,
          .pipeline_guard = pipeline_guard_,
          .scheduler      = scheduler_,
          .base_task      = &base_task_,
          .state          = &state_,
      },
      EditorRenderCoordinator::Callbacks{
          .viewer       = [this]() { return viewer_; },
          .spinner      = [this]() { return spinner_; },
          .active_panel = [this]() { return active_panel_; },
          .needs_full_frame_preview_after_geometry_commit =
              [this]() { return frame_manager_.NeedsFullFramePreviewAfterGeometryCommit(); },
          .apply_state_to_pipeline =
              [this](const AdjustmentState& render_state) { ApplyStateToPipeline(render_state); },
          .refresh_color_temp_runtime_state =
              [this]() { return RefreshColorTempRuntimeStateFromGlobalParams(); },
          .sync_color_temp_controls = [this]() { SyncColorTempControlsFromState(); },
      });
  adjustment_session_ = std::make_unique<EditorAdjustmentSession>(
      EditorAdjustmentSession::Dependencies{
          .pipeline_guard = pipeline_guard_,
          .working_version =
              history_coordinator_ ? &history_coordinator_->WorkingVersion() : nullptr,
          .state           = &state_,
          .committed_state = &committed_state_,
      },
      EditorAdjustmentSession::Callbacks{
          .schedule_quality_preview   = [this]() { ScheduleQualityPreviewRenderFromPipeline(); },
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
  BuildGeometryPanel();
  BuildRawDecodePanel();
  BuildVersioningPanel();
  if (history_coordinator_) {
    history_coordinator_->SetUiContext(versioning::VersionUiContext{
        .version_status     = version_status_,
        .commit_version_btn = commit_version_btn_,
        .undo_tx_btn        = undo_tx_btn_,
        .working_mode_combo = working_mode_combo_,
        .version_log        = version_log_,
        .tx_stack           = tx_stack_,
    });
  }

  shortcut_registry_ = std::make_unique<ShortcutRegistry>(this);
  RegisterShortcuts();

  AppTheme::ApplyFontsRecursively(this);

  UpdateVersionUi();

  frame_manager_.SetViewer(viewer_);
  frame_manager_.SetScopePanel(scope_panel_);
  if (scope_panel_) {
    scope_panel_->SetNeedsRenderCallback([this]() {
      RequestRender(/*use_viewport_region=*/true,
                    /*bump_preview_generation=*/false);
    });
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
  if (viewer_ && geometry_panel_) {
    QObject::connect(viewer_, &QtEditViewer::CropOverlayRectChanged, geometry_panel_,
                     [this](float x, float y, float w, float h, bool /*is_final*/) {
                       if (geometry_panel_) {
                         geometry_panel_->SetCropRectFromViewer(x, y, w, h);
                       }
                     });
    QObject::connect(viewer_, &QtEditViewer::CropOverlayRotationChanged, geometry_panel_,
                     [this](float angle_degrees, bool /*is_final*/) {
                       if (geometry_panel_) {
                         geometry_panel_->SetRotationFromViewer(angle_degrees);
                       }
                     });
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
      .on_trigger =
          [this]() {
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
      .on_trigger =
          [this]() {
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
      .on_trigger =
          [this]() {
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
  if (geometry_panel_ && geometry_panel_->ResetButton()) {
    geometry_panel_->ResetButton()->setToolTip(
        shortcut_registry_->DecorateTooltip(Tr("Reset crop & rotation"), kShortcutResetGeometryId));
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
    if (dynamic_cast<ToneCurveWidget*>(obj) != nullptr && curve_reset_callback_) {
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

  const int right_width =
      std::clamp(static_cast<int>(std::lround(static_cast<double>(available_width) * 0.25)),
                 controls_panel->minimumWidth(), controls_panel->maximumWidth());
  const int center_width = std::max(400, available_width - kVersioningCollapsedWidth - right_width);

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
  if (geometry_panel_) {
    geometry_panel_->RetranslateUi();
  }
  if (drt_panel_) {
    drt_panel_->RetranslateUi();
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
  if (tone_panel_) {
    if (auto* unsupported_label = tone_panel_->ColorTempUnsupportedLabel()) {
      unsupported_label->setText(Tr("Color temperature/tint is unavailable for this image."));
    }
    tone_panel_->RetranslateColorTempModeCombo();
  }
  if (working_mode_combo_) {
    const int  current_value = working_mode_combo_->currentData().toInt();
    const bool prev_sync     = syncing_controls_;
    syncing_controls_        = true;
    working_mode_combo_->clear();
    working_mode_combo_->addItem(Tr("Plain"), static_cast<int>(WorkingMode::Plain));
    working_mode_combo_->addItem(Tr("Incremental"), static_cast<int>(WorkingMode::Incremental));
    const int index = working_mode_combo_->findData(current_value);
    working_mode_combo_->setCurrentIndex(std::max(0, index));
    syncing_controls_ = prev_sync;
  }
  if (lut_browser_widget_) {
    lut_browser_widget_->RetranslateUi();
    RefreshLutBrowserUi();
  }

  if (raw_panel_) {
    raw_panel_->RetranslateUi();
  }
  RefreshHlsTargetUi();
  UpdateViewerZoomLabel(viewer_ ? viewer_->GetViewZoom() : 1.0f);
  RefreshVersioningCollapseUi();
  UpdateVersionUi();
}
void EditorDialog::BuildToneControlPanel() {
  if (!tone_panel_ || !controls_layout_) {
    return;
  }

  const auto default_lut_path           = lut_controller_.DefaultLutPath();

  // If the pipeline already has operator params (loaded from PipelineService/storage),
  // initialize UI state from those params rather than overwriting them.
  const bool loaded_state_from_pipeline = LoadStateFromPipelineIfPresent();
  if (!loaded_state_from_pipeline) {
    // Demo-friendly default: apply a LUT only for brand-new pipelines with no saved params.
    state_.lut_path_ = default_lut_path;
    UpdateAllCdlWheelDerivedColors(state_);
  }
  committed_state_ = state_;

  // Seed a working version from the latest committed one (if any).
  if (history_coordinator_) {
    history_coordinator_->SeedWorkingVersionFromLatest();
  }
  WireLookControlPanel();

  ToneControlPanelWidget::Dependencies deps{
      .session                = adjustment_session_.get(),
      .panel_layout           = controls_layout_,
      .dialog_state           = &state_,
      .dialog_committed_state = &committed_state_,
  };

  ToneControlPanelWidget::Callbacks callbacks{
      .is_global_syncing = [this]() { return syncing_controls_; },
      .request_render    = [this]() { RequestRender(); },
      .register_slider_reset =
          [this](QSlider* slider, std::function<void()> on_reset) {
            RegisterSliderReset(slider, std::move(on_reset));
          },
      .register_curve_reset =
          [this](ToneCurveWidget* widget, std::function<void()> on_reset) {
            RegisterCurveReset(widget, std::move(on_reset));
          },
      .default_adjustment_state = [this]() -> const AdjustmentState& {
        return DefaultAdjustmentState();
      },
      .sync_controls_from_state     = [this]() { SyncControlsFromState(); },
      .prime_color_temp_for_as_shot = [this]() { PrimeColorTempDisplayForAsShot(); },
      .reset_color_temp_to_as_shot  = [this]() { ResetColorTempToAsShot(); },
  };

  tone_panel_->Configure(std::move(deps), std::move(callbacks));
  tone_panel_->Build();
}

void EditorDialog::BuildDisplayTransformPanel() {
  if (!drt_panel_ || !drt_controls_layout_) {
    return;
  }

  DisplayTransformPanelWidget::Dependencies deps{
      .session                = adjustment_session_.get(),
      .panel_layout           = drt_controls_layout_,
      .dialog_state           = &state_,
      .dialog_committed_state = &committed_state_,
  };

  DisplayTransformPanelWidget::Callbacks callbacks{
      .is_global_syncing = [this]() { return syncing_controls_; },
      .request_render    = [this]() { RequestRender(); },
      .register_slider_reset =
          [this](QSlider* slider, std::function<void()> on_reset) {
            RegisterSliderReset(slider, std::move(on_reset));
          },
      .default_adjustment_state = [this]() -> const AdjustmentState& {
        return DefaultAdjustmentState();
      },
      .sync_display_encoding =
          [this](ColorUtils::ColorSpace encoding_space, ColorUtils::EOTF encoding_eotf) {
            frame_manager_.SyncViewerDisplayEncoding(encoding_space, encoding_eotf);
          },
      .load_from_pipeline =
          [this](const DisplayTransformAdjustmentState& base)
          -> std::optional<DisplayTransformAdjustmentState> {
        if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
          return std::nullopt;
        }
        const auto loaded = DisplayTransformPipelineAdapter::Load(*pipeline_guard_->pipeline_, base);
        if (!loaded.loaded_any) {
          return std::nullopt;
        }
        return loaded.state;
      },
  };

  drt_panel_->Configure(std::move(deps), std::move(callbacks));
  drt_panel_->Build();
}

void EditorDialog::BuildGeometryPanel() {
  if (!geometry_panel_ || !geometry_controls_layout_) {
    return;
  }

  GeometryPanelWidget::Dependencies deps{
      .session                = adjustment_session_.get(),
      .panel_layout           = geometry_controls_layout_,
      .dialog_state           = &state_,
      .dialog_committed_state = &committed_state_,
  };

  GeometryPanelWidget::Callbacks callbacks{
      .is_global_syncing = [this]() { return syncing_controls_; },
      .request_render    = [this]() { RequestRender(); },
      .register_slider_reset =
          [this](QSlider* slider, std::function<void()> on_reset) {
            RegisterSliderReset(slider, std::move(on_reset));
          },
      .set_crop_overlay_aspect_lock =
          [this](bool locked, float ratio) {
            if (viewer_) {
              viewer_->SetCropOverlayAspectLock(locked, ratio);
            }
          },
      .set_crop_overlay_rect =
          [this](float x, float y, float w, float h) {
            if (viewer_) {
              viewer_->SetCropOverlayRectNormalized(x, y, w, h);
            }
          },
      .set_crop_overlay_rotation =
          [this](float degrees) {
            if (viewer_) {
              viewer_->SetCropOverlayRotationDegrees(degrees);
            }
          },
      .set_crop_overlay_visible =
          [this](bool visible) {
            if (viewer_) {
              viewer_->SetCropOverlayVisible(visible);
            }
          },
      .set_crop_tool_enabled =
          [this](bool enabled) {
            if (viewer_) {
              viewer_->SetCropToolEnabled(enabled);
            }
          },
      .source_aspect_ratio =
          [this]() -> float {
            if (viewer_ && viewer_->GetWidth() > 0 && viewer_->GetHeight() > 0) {
              return static_cast<float>(viewer_->GetWidth()) /
                     static_cast<float>(viewer_->GetHeight());
            }
            return 1.0f;
          },
  };

  geometry_panel_->Configure(std::move(deps), std::move(callbacks));
  geometry_panel_->Build();
}

void EditorDialog::BuildRawDecodePanel() {
  if (!raw_panel_ || !raw_controls_layout_) {
    return;
  }

  RawDecodePanelWidget::Dependencies deps{
      .session                = adjustment_session_.get(),
      .panel_layout           = raw_controls_layout_,
      .dialog_state           = &state_,
      .dialog_committed_state = &committed_state_,
  };

  RawDecodePanelWidget::Callbacks callbacks{
      .is_global_syncing = [this]() { return syncing_controls_; },
      .request_render    = [this]() { RequestRender(); },
      .load_from_pipeline =
          [this](const RawDecodeAdjustmentState& base) -> std::optional<RawDecodeAdjustmentState> {
        if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
          return std::nullopt;
        }
        const auto loaded = RawPipelineAdapter::Load(*pipeline_guard_->pipeline_, base);
        if (!loaded.loaded_any) {
          return std::nullopt;
        }
        return loaded.state;
      },
  };

  raw_panel_->Configure(std::move(deps), std::move(callbacks));
  raw_panel_->Build();
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
