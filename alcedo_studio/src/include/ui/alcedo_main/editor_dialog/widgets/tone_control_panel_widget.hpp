//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QComboBox>
#include <QLabel>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <functional>

#include "ui/alcedo_main/editor_dialog/session/adjustment_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/tone_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_curve_widget.hpp"

namespace alcedo::ui {

class EditorAdjustmentSession;

class ToneControlPanelWidget final : public AdjustmentPanelWidget {
 public:
  struct Dependencies {
    EditorAdjustmentSession* session                = nullptr;
    QVBoxLayout*             panel_layout           = nullptr;
    AdjustmentState*         dialog_state           = nullptr;
    AdjustmentState*         dialog_committed_state = nullptr;
  };

  struct Callbacks {
    std::function<bool()>                                   is_global_syncing;
    std::function<void()>                                   request_render;
    std::function<void(QSlider*, std::function<void()>)>    register_slider_reset;
    std::function<void(ToneCurveWidget*, std::function<void()>)>
                                                            register_curve_reset;
    std::function<const AdjustmentState&()>                 default_adjustment_state;
    std::function<void()>                                   sync_controls_from_state;
    std::function<void()>                                   prime_color_temp_for_as_shot;
    std::function<void()>                                   reset_color_temp_to_as_shot;
  };

  explicit ToneControlPanelWidget(QWidget* parent = nullptr);

  void Configure(Dependencies deps, Callbacks callbacks);
  void Build();

  // AdjustmentPanelWidget API
  auto PanelId() const -> AdjustmentPanelId override { return AdjustmentPanelId::Tone; }
  void LoadFromPipeline() override;
  void ReloadFromCommittedState() override;
  void SetSyncing(bool syncing) override;

  // Sync slider/curve UI from the dialog's legacy AdjustmentState.
  void SyncControlsFromDialogState();
  void SyncColorTempControlsFromDialogState();
  void RetranslateColorTempModeCombo();

  auto ColorTempUnsupportedLabel() const -> QLabel* { return color_temp_unsupported_label_; }
  auto ColorTempCctSlider() const -> QSlider* { return color_temp_cct_slider_; }
  auto ColorTempTintSlider() const -> QSlider* { return color_temp_tint_slider_; }
  auto ColorTempModeCombo() const -> QComboBox* { return color_temp_mode_combo_; }

 private:
  void BuildToneSection();
  void BuildToneCurveSection();
  void BuildColorSection();
  void BuildDetailSection();

  void ProjectToneStateToDialog();
  void PullToneStateFromDialog();
  void PullCommittedToneStateFromDialog();

  auto IsSyncing() const -> bool;
  void RequestPipelineRender();
  auto SessionPreviewParamsForTone(AdjustmentField field) -> nlohmann::json;
  void PreviewToneField(AdjustmentField field);
  void CommitToneField(AdjustmentField field);
  void ResetToneFieldToDefault(AdjustmentField                                    field,
                               const std::function<void(const ToneAdjustmentState&,
                                                        const AdjustmentState&)>& apply_default);
  void ResetCurveToDefaultLocal();
  void PromoteColorTempToCustomForEditing();

  Dependencies deps_{};
  Callbacks    callbacks_{};
  bool         local_syncing_ = false;

  ToneAdjustmentState tone_state_{};
  ToneAdjustmentState committed_tone_state_{};

  QSlider*         exposure_slider_              = nullptr;
  QSlider*         contrast_slider_              = nullptr;
  QSlider*         highlights_slider_            = nullptr;
  QSlider*         shadows_slider_               = nullptr;
  QSlider*         whites_slider_                = nullptr;
  QSlider*         blacks_slider_                = nullptr;
  ToneCurveWidget* curve_widget_                 = nullptr;
  QSlider*         saturation_slider_            = nullptr;
  QComboBox*       color_temp_mode_combo_        = nullptr;
  QSlider*         color_temp_cct_slider_        = nullptr;
  QSlider*         color_temp_tint_slider_       = nullptr;
  QLabel*          color_temp_unsupported_label_ = nullptr;
  QSlider*         sharpen_slider_               = nullptr;
  QSlider*         clarity_slider_               = nullptr;
};

}  // namespace alcedo::ui
