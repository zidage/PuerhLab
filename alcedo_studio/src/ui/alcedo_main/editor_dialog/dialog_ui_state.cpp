#include "ui/alcedo_main/editor_dialog/dialog_internal.hpp"

namespace alcedo::ui {

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

void EditorDialog::SyncControlsFromState() {
  if (!controls_) {
    return;
  }

  syncing_controls_ = true;
  LoadActiveHlsProfile(state_);
  SanitizeOdtStateForUi(state_.odt_);

  if (raw_panel_) {
    raw_panel_->SyncControlsFromDialogState();
  }
  if (look_panel_) {
    look_panel_->SyncControlsFromDialogState();
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
  syncing_controls_ = false;
}
}  // namespace alcedo::ui
