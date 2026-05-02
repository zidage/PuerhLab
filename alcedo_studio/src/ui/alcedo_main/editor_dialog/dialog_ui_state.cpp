#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

void EditorDialog::RefreshHlsTargetUi() {
  if (!hls_target_label_ && hls_candidate_buttons_.empty()) {
    return;
  }

  const float hue = WrapHueDegrees(state_.hls_target_hue_);
  if (hls_target_label_) {
    hls_target_label_->setText(Tr("Target Hue: %1 deg").arg(hue, 0, 'f', 0));
  }

  const int selected_idx = ClosestHlsCandidateHueIndex(hue);
  for (int i = 0; i < static_cast<int>(hls_candidate_buttons_.size()); ++i) {
    auto* btn = hls_candidate_buttons_[i];
    if (!btn) {
      continue;
    }
    const bool    selected    = (i == selected_idx);
    const QColor  swatch      = HlsCandidateColor(kHlsCandidateHues[static_cast<size_t>(i)]);
    const auto    border_w_px = selected ? "3px" : "1px";
    const QString border_col  = selected
                                    ? AppTheme::Instance().accentColor().name(QColor::HexRgb)
                                    : AppTheme::Instance().glassStrokeColor().name(QColor::HexArgb);
    btn->setToolTip(Tr("Hue %1 deg").arg(kHlsCandidateHues[static_cast<size_t>(i)], 0, 'f', 0));
    btn->setStyleSheet(QString("QPushButton {"
                               "  background: %1;"
                               "  border: %2 solid %3;"
                               "  border-radius: 11px;"
                               "}"
                               "QPushButton:hover {"
                               "  border-color: %4;"
                               "}")
                           .arg(swatch.name(QColor::HexRgb), border_w_px, border_col,
                                AppTheme::Instance().accentSecondaryColor().name(QColor::HexRgb)));
  }
}

void EditorDialog::RefreshCdlOffsetLabels() {
  if (lift_offset_label_) {
    lift_offset_label_->setText(FormatWheelDeltaText(state_.lift_wheel_, false));
  }
  if (gamma_offset_label_) {
    gamma_offset_label_->setText(FormatWheelDeltaText(state_.gamma_wheel_, true));
  }
  if (gain_offset_label_) {
    gain_offset_label_->setText(FormatWheelDeltaText(state_.gain_wheel_, true));
  }
}

auto EditorDialog::DefaultAdjustmentState() -> const AdjustmentState& {
  static const AdjustmentState defaults{};
  return defaults;
}

void EditorDialog::CacheAsShotColorTemp(float cct, float tint) {
  last_known_as_shot_cct_ =
      std::clamp(cct, static_cast<float>(kColorTempCctMin), static_cast<float>(kColorTempCctMax));
  last_known_as_shot_tint_           = std::clamp(tint, static_cast<float>(kColorTempTintMin),
                                                  static_cast<float>(kColorTempTintMax));
  has_last_known_as_shot_color_temp_ = true;
}

void EditorDialog::PrimeColorTempDisplayForAsShot() {
  if (!has_last_known_as_shot_color_temp_) {
    return;
  }
  state_.color_temp_resolved_cct_            = last_known_as_shot_cct_;
  state_.color_temp_resolved_tint_           = last_known_as_shot_tint_;
  committed_state_.color_temp_resolved_cct_  = last_known_as_shot_cct_;
  committed_state_.color_temp_resolved_tint_ = last_known_as_shot_tint_;
}

void EditorDialog::WarmAsShotColorTempCacheFromRawMetadata() {
  if (has_last_known_as_shot_color_temp_ || !pipeline_guard_ || !pipeline_guard_->pipeline_) {
    return;
  }

  auto params = pipeline_guard_->pipeline_->GetGlobalParams();
  try {
    static const nlohmann::json kAsShotColorTempParams = {
        {"color_temp", {{"mode", "as_shot"}, {"cct", 6500.0f}, {"tint", 0.0f}}}};
    ColorTempOp as_shot_probe(kAsShotColorTempParams);
    as_shot_probe.SetGlobalParams(params);
  } catch (...) {
    return;
  }

  if (!params.color_temp_matrices_valid_) {
    return;
  }

  CacheAsShotColorTemp(params.color_temp_resolved_cct_, params.color_temp_resolved_tint_);
  if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT) {
    PrimeColorTempDisplayForAsShot();
  }
}

void EditorDialog::RegisterSliderReset(QSlider* slider, std::function<void()> on_reset) {
  if (!slider || !on_reset) {
    return;
  }
  slider->installEventFilter(this);
  slider_reset_callbacks_[slider] = std::move(on_reset);
}

void EditorDialog::RegisterCurveReset(ToneCurveWidget* widget, std::function<void()> on_reset) {
  if (!widget || !on_reset) {
    return;
  }
  widget->installEventFilter(this);
  curve_reset_callback_ = std::move(on_reset);
}

void EditorDialog::ResetFieldToDefault(
    AdjustmentField field, const std::function<void(const AdjustmentState&)>& apply_default) {
  if (!apply_default) {
    return;
  }
  const auto& defaults = DefaultAdjustmentState();
  apply_default(defaults);
  SyncControlsFromState();
  RequestRender();
  CommitAdjustment(field);
}

void EditorDialog::ResetColorTempToAsShot() {
  if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT) {
    return;
  }
  state_.color_temp_mode_ = ColorTempMode::AS_SHOT;
  PrimeColorTempDisplayForAsShot();
  SyncColorTempControlsFromState();
  RequestRender();
  CommitAdjustment(AdjustmentField::ColorTemp);
}

void EditorDialog::ResetCurveToDefault() {
  state_.curve_points_ = DefaultCurveControlPoints();
  SyncControlsFromState();
  RequestRender();
  CommitAdjustment(AdjustmentField::Curve);
}

void EditorDialog::UpdateViewerZoomLabel(float zoom) {
  if (!viewer_zoom_value_label_) {
    return;
  }
  const float clamped = std::max(1.0f, zoom);
  viewer_zoom_value_label_->setText(Tr("%1%").arg(clamped * 100.0f, 0, 'f', 0));

  if (viewer_zoom_resolution_label_) {
    const uint32_t w = exif_display_.width_;
    const uint32_t h = exif_display_.height_;
    if (w > 0 && h > 0) {
      viewer_zoom_resolution_label_->setText(Tr("%1 × %2 px").arg(w).arg(h));
    } else {
      viewer_zoom_resolution_label_->setText(QStringLiteral("-- × -- px"));
    }
  }
}

void EditorDialog::RefreshLutBrowserUi() {
  if (!lut_browser_widget_) {
    return;
  }
  const auto view_model = lut_controller_.Refresh(state_.lut_path_, false);
  lut_browser_widget_->SetDirectoryInfo(view_model.directory_text_, view_model.status_text_,
                                        view_model.can_open_directory_);
  lut_browser_widget_->SetEntries(view_model.entries_, view_model.selected_path_);
}

void EditorDialog::ForceRefreshLutBrowserUi() {
  if (!lut_browser_widget_) {
    return;
  }
  const auto view_model = lut_controller_.Refresh(state_.lut_path_, true);
  lut_browser_widget_->SetDirectoryInfo(view_model.directory_text_, view_model.status_text_,
                                        view_model.can_open_directory_);
  lut_browser_widget_->SetEntries(view_model.entries_, view_model.selected_path_);
}

void EditorDialog::OpenLutFolder() {
  const auto&     directory = lut_controller_.directory();
  std::error_code ec;
  if (directory.empty() || !std::filesystem::is_directory(directory, ec) || ec) {
    QMessageBox::warning(this, Tr("LUT"), Tr("LUT folder is unavailable."));
    return;
  }
  if (!QDesktopServices::openUrl(
          QUrl::fromLocalFile(QString::fromStdWString(directory.wstring())))) {
    QMessageBox::warning(this, Tr("LUT"), Tr("Failed to open the LUT folder."));
  }
}

void EditorDialog::RefreshPanelSwitchUi() {
  if (!tone_panel_btn_ || !look_panel_btn_ || !drt_panel_btn_ || !geometry_panel_btn_ ||
      !raw_panel_btn_) {
    return;
  }
  const bool tone_active     = (active_panel_ == ControlPanelKind::Tone);
  const bool look_active     = (active_panel_ == ControlPanelKind::Look);
  const bool drt_active      = (active_panel_ == ControlPanelKind::DisplayRenderingTransform);
  const bool geometry_active = (active_panel_ == ControlPanelKind::Geometry);
  const bool raw_active      = (active_panel_ == ControlPanelKind::RawDecode);

  const auto apply_panel_button_state = [](QPushButton* button, bool active, bool is_first,
                                           bool is_last) {
    if (!button) {
      return;
    }

    button->setChecked(active);
    button->setIcon(RenderPanelToggleIcon(
        button->property(kPanelIconPathProperty).toString(),
        active ? AppTheme::Instance().bgCanvasColor() : AppTheme::Instance().textColor(),
        kPanelToggleIconSize, button->devicePixelRatioF()));
    button->setStyleSheet(AppTheme::EditorPanelToggleStyle(active, is_first, is_last));
  };

  apply_panel_button_state(tone_panel_btn_, tone_active, true, false);
  apply_panel_button_state(look_panel_btn_, look_active, false, false);
  apply_panel_button_state(drt_panel_btn_, drt_active, false, false);
  apply_panel_button_state(geometry_panel_btn_, geometry_active, false, false);
  apply_panel_button_state(raw_panel_btn_, raw_active, false, true);
}

void EditorDialog::RefreshVersioningCollapseUi() {
  if (!versioning_panel_host_ || !versioning_collapsed_nav_) {
    return;
  }

  const auto& theme = AppTheme::Instance();
  const qreal progress =
      std::clamp(versioning_panel_progress_, static_cast<qreal>(0.0), static_cast<qreal>(1.0));
  if (versioning_pages_stack_) {
    versioning_pages_stack_->setCurrentIndex(static_cast<int>(versioning_active_page_));
  }

  // Show or hide the flyout based on animation progress.
  if (versioning_flyout_) {
    if (progress > 0.0) {
      if (!versioning_flyout_->isVisible()) {
        versioning_flyout_->show();
        versioning_flyout_->raise();
        RepositionVersioningFlyout();
        QTimer::singleShot(0, this, [this]() {
          if (!versioning_flyout_ || versioning_panel_progress_ <= 0.0) {
            return;
          }
          RepositionVersioningFlyout();
        });
      }
    } else {
      if (versioning_flyout_->isVisible()) {
        versioning_flyout_->hide();
      }
    }
  }

  if (versioning_panel_opacity_effect_) {
    versioning_panel_opacity_effect_->setOpacity(progress);
  }

  const bool panel_expanded    = progress >= 0.5;
  versioning_collapsed_        = !panel_expanded;

  const auto update_nav_button = [&](QPushButton* button, const QString& icon_path,
                                     const QString& label, VersioningFlyoutPage page) {
    if (!button) {
      return;
    }
    const bool active = panel_expanded && versioning_active_page_ == page;
    button->setProperty("versioningActive", active);
    button->style()->unpolish(button);
    button->style()->polish(button);
    button->setIcon(RenderPanelToggleIcon(icon_path,
                                          active ? theme.textColor() : theme.textMutedColor(),
                                          kVersioningRailIconSize, button->devicePixelRatioF()));
    const QString tooltip = active ? Tr("Hide %1").arg(label) : Tr("Show %1").arg(label);
    button->setToolTip(tooltip);
    button->setAccessibleName(tooltip);
  };

  update_nav_button(versioning_history_btn_,
                    QStringLiteral(":/history_icons/git-commit-horizontal.svg"), Tr("Edit History"),
                    VersioningFlyoutPage::History);
  update_nav_button(versioning_versions_btn_, QStringLiteral(":/panel_icons/git-branch.svg"),
                    Tr("Version Tree"), VersioningFlyoutPage::Versions);
}

void EditorDialog::SetVersioningCollapsed(bool collapsed, bool animate) {
  const qreal target_progress = collapsed ? 0.0 : 1.0;
  if (!animate || std::abs(versioning_panel_progress_ - target_progress) < 0.001) {
    if (versioning_panel_anim_) {
      versioning_panel_anim_->stop();
    }
    versioning_panel_progress_ = target_progress;
    versioning_collapsed_      = collapsed;
    if (!collapsed && versioning_pages_stack_) {
      versioning_pages_stack_->setCurrentIndex(static_cast<int>(versioning_active_page_));
    }
    RefreshVersioningCollapseUi();
    return;
  }

  // Show flyout immediately at the start of an expand animation so the
  // opacity fade has something to render into.
  if (!collapsed && versioning_pages_stack_) {
    versioning_pages_stack_->setCurrentIndex(static_cast<int>(versioning_active_page_));
  }
  if (!collapsed && versioning_flyout_ && !versioning_flyout_->isVisible()) {
    versioning_flyout_->show();
    versioning_flyout_->raise();
    RepositionVersioningFlyout();
    QTimer::singleShot(0, this, [this]() {
      if (!versioning_flyout_ || versioning_panel_progress_ <= 0.0) {
        return;
      }
      RepositionVersioningFlyout();
    });
  }

  if (!versioning_panel_anim_) {
    versioning_panel_anim_ = new QVariantAnimation(this);
    versioning_panel_anim_->setDuration(kVersioningAnimationMs);
    versioning_panel_anim_->setEasingCurve(QEasingCurve::OutCubic);
    QObject::connect(versioning_panel_anim_, &QVariantAnimation::valueChanged, this,
                     [this](const QVariant& value) {
                       versioning_panel_progress_ = value.toReal();
                       RefreshVersioningCollapseUi();
                     });
    QObject::connect(versioning_panel_anim_, &QVariantAnimation::finished, this, [this]() {
      if (!versioning_panel_anim_) {
        return;
      }
      versioning_panel_progress_ = versioning_panel_anim_->endValue().toReal();
      versioning_collapsed_      = versioning_panel_progress_ < 0.5;
      RefreshVersioningCollapseUi();
    });
  }

  versioning_panel_anim_->stop();
  versioning_panel_anim_->setStartValue(versioning_panel_progress_);
  versioning_panel_anim_->setEndValue(target_progress);
  versioning_panel_anim_->start();
}

void EditorDialog::RepositionVersioningFlyout() {
  if (!versioning_flyout_ || !viewer_container_) {
    return;
  }

  versioning_flyout_->ensurePolished();
  if (auto* layout = versioning_flyout_->layout()) {
    layout->activate();
  }
  if (versioning_pages_stack_) {
    versioning_pages_stack_->ensurePolished();
    if (auto* layout = versioning_pages_stack_->layout()) {
      layout->activate();
    }
    if (auto* page = versioning_pages_stack_->currentWidget()) {
      page->ensurePolished();
      if (auto* layout = page->layout()) {
        layout->activate();
      }
    }
  }
  if (shared_versioning_root_) {
    shared_versioning_root_->ensurePolished();
    if (auto* layout = shared_versioning_root_->layout()) {
      layout->activate();
    }
  }

  const QRect viewer_rect = viewer_container_->geometry().adjusted(14, 14, -14, -14);
  const int   gap         = 14;
  int         flyout_x    = viewer_rect.left() + 4;
  if (versioning_panel_host_) {
    flyout_x = std::max(flyout_x, versioning_panel_host_->geometry().right() + gap);
  }
  flyout_x                  = std::max(flyout_x, kEditorOuterMargin + 4);

  const int available_width = std::max(0, viewer_rect.right() - flyout_x - 12);
  const int desired_width =
      std::clamp(static_cast<int>(std::lround(static_cast<double>(viewer_rect.width()) * 0.30)),
                 kVersioningExpandedMinWidth, kVersioningExpandedMaxWidth);
  const int flyout_y = viewer_rect.top() + 2;
  const int flyout_w = std::clamp(
      desired_width, std::min(kVersioningExpandedMinWidth, std::max(220, available_width)),
      std::max(220, available_width));
  int content_height = kVersioningExpandedMinHeight;
  if (versioning_pages_stack_) {
    if (auto* page = versioning_pages_stack_->currentWidget()) {
      page->ensurePolished();
      content_height = std::max(content_height, page->sizeHint().height() + 32);
    } else {
      content_height = std::max(content_height, versioning_pages_stack_->sizeHint().height() + 32);
    }
  } else if (shared_versioning_root_) {
    content_height = std::max(content_height, shared_versioning_root_->sizeHint().height());
  }
  const int flyout_h =
      std::clamp(content_height, kVersioningExpandedMinHeight,
                 std::min(kVersioningExpandedMaxHeight, std::max(220, viewer_rect.height() - 20)));

  versioning_flyout_->setGeometry(flyout_x, flyout_y, flyout_w, flyout_h);
  if (auto* layout = versioning_flyout_->layout()) {
    layout->activate();
  }
  if (shared_versioning_root_) {
    if (auto* layout = shared_versioning_root_->layout()) {
      layout->activate();
    }
  }
  if (shared_versioning_root_) {
    const QRect rect = shared_versioning_root_->rect();
    if (rect.isValid() && rect.width() > 0 && rect.height() > 0) {
      QPainterPath path;
      path.addRoundedRect(QRectF(rect), 14.0, 14.0);
      shared_versioning_root_->setMask(QRegion(path.toFillPolygon().toPolygon()));
    } else {
      shared_versioning_root_->clearMask();
    }
  }
}

void EditorDialog::SetActiveControlPanel(ControlPanelKind panel) {
  const ControlPanelKind previous_panel = active_panel_;
  active_panel_                         = panel;
  if (control_panels_stack_) {
    int panel_index = 0;
    if (panel == ControlPanelKind::Look) {
      panel_index = 1;
    } else if (panel == ControlPanelKind::DisplayRenderingTransform) {
      panel_index = 2;
    } else if (panel == ControlPanelKind::Geometry) {
      panel_index = 3;
    } else if (panel == ControlPanelKind::RawDecode) {
      panel_index = 4;
    }
    control_panels_stack_->setCurrentIndex(panel_index);
  }

  const bool geometry_active = (panel == ControlPanelKind::Geometry);

  if (viewer_) {
    if (geometry_panel_) {
      geometry_panel_->SyncControlsFromDialogState();
    }
    viewer_->SetCropOverlayVisible(geometry_active);
    viewer_->SetCropToolEnabled(geometry_active);
  }
  RefreshPanelSwitchUi();
  if (pipeline_initialized_) {
    const bool geometry_transition =
        previous_panel == ControlPanelKind::Geometry || panel == ControlPanelKind::Geometry;
    if (geometry_transition) {
      InvalidateDetailPreviewState();
    }
    RequestRender(frame_manager_.UseViewportRegionForPanelChange(previous_panel, panel),
                  geometry_transition);
    ScheduleQualityPreviewRenderFromPipeline();
  }
}

void EditorDialog::PromoteColorTempToCustomForEditing() {
  if (state_.color_temp_mode_ == ColorTempMode::CUSTOM) {
    return;
  }
  state_.color_temp_custom_cct_  = DisplayedColorTempCct(state_);
  state_.color_temp_custom_tint_ = DisplayedColorTempTint(state_);
  state_.color_temp_mode_        = ColorTempMode::CUSTOM;

  const bool prev_sync           = syncing_controls_;
  syncing_controls_              = true;
  if (tone_panel_) {
    if (auto* combo = tone_panel_->ColorTempModeCombo()) {
      combo->setCurrentIndex(ColorTempModeToComboIndex(state_.color_temp_mode_));
    }
  }
  syncing_controls_ = prev_sync;
}

// Returns true if any resolved color temp value actually changed.
auto EditorDialog::RefreshColorTempRuntimeStateFromGlobalParams() -> bool {
  if (!pipeline_guard_ || !pipeline_guard_->pipeline_) {
    return false;
  }

  const auto& global = pipeline_guard_->pipeline_->GetGlobalParams();
  const float new_cct =
      std::clamp(global.color_temp_resolved_cct_, static_cast<float>(kColorTempCctMin),
                 static_cast<float>(kColorTempCctMax));
  const float new_tint =
      std::clamp(global.color_temp_resolved_tint_, static_cast<float>(kColorTempTintMin),
                 static_cast<float>(kColorTempTintMax));
  const bool new_sup = global.color_temp_matrices_valid_;

  const bool changed = !NearlyEqual(state_.color_temp_resolved_cct_, new_cct) ||
                       !NearlyEqual(state_.color_temp_resolved_tint_, new_tint) ||
                       state_.color_temp_supported_ != new_sup;

  state_.color_temp_resolved_cct_            = new_cct;
  state_.color_temp_resolved_tint_           = new_tint;
  state_.color_temp_supported_               = new_sup;

  committed_state_.color_temp_resolved_cct_  = new_cct;
  committed_state_.color_temp_resolved_tint_ = new_tint;
  committed_state_.color_temp_supported_     = new_sup;
  if (state_.color_temp_mode_ == ColorTempMode::AS_SHOT && new_sup) {
    CacheAsShotColorTemp(new_cct, new_tint);
  }

  return changed;
}

void EditorDialog::SyncColorTempControlsFromState() {
  const bool prev_sync = syncing_controls_;
  syncing_controls_    = true;

  if (tone_panel_) {
    tone_panel_->SyncColorTempControlsFromDialogState();
  }

  syncing_controls_ = prev_sync;
}

void EditorDialog::RefreshVersionLogSelectionStyles() {
  if (!version_log_) {
    return;
  }
  for (int i = 0; i < version_log_->count(); ++i) {
    auto* item = version_log_->item(i);
    if (!item) {
      continue;
    }
    auto* w = version_log_->itemWidget(item);
    if (!w) {
      continue;
    }
    if (auto* card = dynamic_cast<HistoryCardWidget*>(w)) {
      card->SetSelected(item->isSelected());
    }
  }
}

void EditorDialog::SyncControlsFromState() {
  if (!controls_) {
    return;
  }

  syncing_controls_ = true;
  LoadActiveHlsProfile(state_);
  SanitizeOdtStateForUi(state_.odt_);
  RefreshLutBrowserUi();

  if (raw_panel_) {
    raw_panel_->SyncControlsFromDialogState();
  }
  if (lift_disc_widget_) {
    lift_disc_widget_->SetPosition(state_.lift_wheel_.disc_position_);
  }
  if (gamma_disc_widget_) {
    gamma_disc_widget_->SetPosition(state_.gamma_wheel_.disc_position_);
  }
  if (gain_disc_widget_) {
    gain_disc_widget_->SetPosition(state_.gain_wheel_.disc_position_);
  }
  if (lift_master_slider_) {
    lift_master_slider_->setValue(CdlMasterToSliderUi(state_.lift_wheel_.master_offset_));
  }
  if (gamma_master_slider_) {
    gamma_master_slider_->setValue(CdlMasterToSliderUi(-state_.gamma_wheel_.master_offset_));
  }
  if (gain_master_slider_) {
    gain_master_slider_->setValue(CdlMasterToSliderUi(state_.gain_wheel_.master_offset_));
  }
  if (hls_hue_adjust_slider_) {
    hls_hue_adjust_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_adjust_)));
  }
  if (hls_lightness_adjust_slider_) {
    hls_lightness_adjust_slider_->setValue(
        static_cast<int>(std::lround(state_.hls_lightness_adjust_)));
  }
  if (hls_saturation_adjust_slider_) {
    hls_saturation_adjust_slider_->setValue(
        static_cast<int>(std::lround(state_.hls_saturation_adjust_)));
  }
  if (hls_hue_range_slider_) {
    hls_hue_range_slider_->setValue(static_cast<int>(std::lround(state_.hls_hue_range_)));
  }
  if (tone_panel_) {
    tone_panel_->ReloadFromCommittedState();
  }
  if (drt_panel_) {
    drt_panel_->SyncControlsFromDialogState();
  }
  if (geometry_panel_) {
    geometry_panel_->SyncControlsFromDialogState();
  }
  if (viewer_) {
    frame_manager_.SyncViewerDisplayEncoding(state_.odt_.encoding_space_,
                                             state_.odt_.encoding_eotf_);
    const bool geometry_active = (active_panel_ == ControlPanelKind::Geometry);
    viewer_->SetCropOverlayVisible(geometry_active);
    viewer_->SetCropToolEnabled(geometry_active);
  }
  RefreshHlsTargetUi();
  RefreshCdlOffsetLabels();

  syncing_controls_ = false;
}
}  // namespace alcedo::ui
