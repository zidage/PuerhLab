//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QComboBox>
#include <QEvent>
#include <QLabel>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <functional>
#include <map>
#include <optional>

#include "ui/alcedo_main/editor_dialog/session/adjustment_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/color_temp_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/state/tone_adjustment_state.hpp"
#include "ui/alcedo_main/editor_dialog/widgets/tone_curve_widget.hpp"

namespace alcedo {
class CPUPipelineExecutor;
}

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
    std::function<bool()>                   is_global_syncing;
    std::function<void()>                   request_render;
    std::function<const AdjustmentState&()> default_adjustment_state;
    std::function<void()>                   sync_controls_from_state;
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
  void CacheAsShotColorTemp(float cct, float tint);
  void PrimeColorTempDisplayForAsShot();
  void WarmAsShotColorTempCacheFromPipeline(CPUPipelineExecutor* pipeline);
  auto RefreshColorTempRuntimeStateFromGlobalParams(CPUPipelineExecutor* pipeline) -> bool;
  void ResetColorTempToAsShot();
  void ClearSubmittedColorTempRequest();
  void MarkSubmittedColorTempRequest(const AdjustmentState& state);
  auto ShouldSubmitColorTempRequest(bool                   operator_missing,
                                    const AdjustmentState& render_state) const -> bool;

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
  void ProjectColorTempStateToDialog();
  void PullColorTempStateFromDialog();
  void PullCommittedColorTempStateFromDialog();

  auto IsSyncing() const -> bool;
  bool eventFilter(QObject* obj, QEvent* event) override;
  void RegisterSliderReset(QSlider* slider, std::function<void()> on_reset);
  void RegisterCurveReset(ToneCurveWidget* widget, std::function<void()> on_reset);
  void RequestPipelineRender();
  auto SessionPreviewParamsForTone(AdjustmentField field) -> nlohmann::json;
  void PreviewToneField(AdjustmentField field);
  void CommitToneField(AdjustmentField field);
  void ResetToneFieldToDefault(
      AdjustmentField                                                                field,
      const std::function<void(const ToneAdjustmentState&, const AdjustmentState&)>& apply_default);
  void                                      ResetCurveToDefaultLocal();
  void                                      PromoteColorTempToCustomForEditing();

  Dependencies                              deps_{};
  Callbacks                                 callbacks_{};
  bool                                      local_syncing_ = false;

  ToneAdjustmentState                       tone_state_{};
  ToneAdjustmentState                       committed_tone_state_{};
  ColorTempAdjustmentState                  color_temp_state_{};
  ColorTempAdjustmentState                  committed_color_temp_state_{};
  std::optional<ColorTempRequestSnapshot>   last_submitted_color_temp_request_{};
  float                                     last_known_as_shot_cct_            = 6500.0f;
  float                                     last_known_as_shot_tint_           = 0.0f;
  bool                                      has_last_known_as_shot_color_temp_ = false;
  std::map<QSlider*, std::function<void()>> slider_reset_callbacks_{};
  std::function<void()>                     curve_reset_callback_{};

  QSlider*                                  exposure_slider_              = nullptr;
  QSlider*                                  contrast_slider_              = nullptr;
  QSlider*                                  highlights_slider_            = nullptr;
  QSlider*                                  shadows_slider_               = nullptr;
  QSlider*                                  whites_slider_                = nullptr;
  QSlider*                                  blacks_slider_                = nullptr;
  ToneCurveWidget*                          curve_widget_                 = nullptr;
  QSlider*                                  saturation_slider_            = nullptr;
  QComboBox*                                color_temp_mode_combo_        = nullptr;
  QSlider*                                  color_temp_cct_slider_        = nullptr;
  QSlider*                                  color_temp_tint_slider_       = nullptr;
  QLabel*                                   color_temp_unsupported_label_ = nullptr;
  QSlider*                                  sharpen_slider_               = nullptr;
  QSlider*                                  clarity_slider_               = nullptr;
};

}  // namespace alcedo::ui
