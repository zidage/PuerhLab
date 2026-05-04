//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFrame>
#include <QPushButton>
#include <QSlider>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "ui/alcedo_main/editor_dialog/session/adjustment_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/display_transform_adjustment_state.hpp"

namespace alcedo::ui {

class EditorAdjustmentSession;

class DisplayTransformPanelWidget final : public AdjustmentPanelWidget {
 public:
  struct Dependencies {
    EditorAdjustmentSession* session                = nullptr;
    QVBoxLayout*             panel_layout           = nullptr;
    AdjustmentState*         dialog_state           = nullptr;
    AdjustmentState*         dialog_committed_state = nullptr;
  };

  struct Callbacks {
    std::function<bool()>                                         is_global_syncing;
    std::function<void()>                                         request_render;
    std::function<const AdjustmentState&()>                       default_adjustment_state;
    std::function<void(ColorUtils::ColorSpace, ColorUtils::EOTF)> sync_display_encoding;
    std::function<std::optional<DisplayTransformAdjustmentState>(
        const DisplayTransformAdjustmentState&)>
        load_from_pipeline;
  };

  explicit DisplayTransformPanelWidget(QWidget* parent = nullptr);

  void Configure(Dependencies deps, Callbacks callbacks);
  void Build();

  auto PanelId() const -> AdjustmentPanelId override { return AdjustmentPanelId::DisplayTransform; }
  void LoadFromPipeline() override;
  void ReloadFromCommittedState() override;
  void SetSyncing(bool syncing) override;

  void SyncControlsFromDialogState();
  void RetranslateUi();

 private:
  struct OpenDrtDetailSliderBinding {
    QSlider*                                              slider_ = nullptr;
    QDoubleSpinBox*                                       spin_   = nullptr;
    float                                                 min_    = 0.0f;
    float                                                 max_    = 1.0f;
    float                                                 scale_  = 100.0f;
    std::function<float(const odt_cpu::OpenDRTSettings&)> getter_{};
  };

  void         ProjectDisplayTransformStateToDialog();
  void         PullDisplayTransformStateFromDialog();
  void         PullCommittedDisplayTransformStateFromDialog();

  auto         IsSyncing() const -> bool;
  bool         eventFilter(QObject* obj, QEvent* event) override;
  void         RegisterSliderReset(QSlider* slider, std::function<void()> on_reset);
  void         RequestPipelineRender();
  void         SyncViewerDisplayEncoding();
  void         PreviewOdtField();
  void         CommitOdtField();
  void         ResetOdtFieldToDefault(const std::function<void(DisplayTransformAdjustmentState&,
                                                       const AdjustmentState&)>& apply_default);

  void         RefreshOdtMethodUi();
  void         RefreshOdtEncodingEotfComboFromState();
  void         SyncOpenDrtDetailControlsFromState();
  void         MarkOpenDrtLookPresetCustomForEditing();
  void         MarkOpenDrtTonescalePresetCustomForEditing();

  Dependencies deps_{};
  Callbacks    callbacks_{};
  bool         local_syncing_ = false;

  DisplayTransformAdjustmentState           display_state_{};
  DisplayTransformAdjustmentState           committed_display_state_{};

  QComboBox*                                odt_encoding_space_combo_            = nullptr;
  QComboBox*                                odt_encoding_eotf_combo_             = nullptr;
  QSlider*                                  odt_peak_luminance_slider_           = nullptr;
  QPushButton*                              odt_aces_method_card_                = nullptr;
  QPushButton*                              odt_open_drt_method_card_            = nullptr;
  QStackedWidget*                           odt_method_stack_                    = nullptr;
  QComboBox*                                odt_aces_limiting_space_combo_       = nullptr;
  QComboBox*                                odt_open_drt_look_preset_combo_      = nullptr;
  QComboBox*                                odt_open_drt_tonescale_preset_combo_ = nullptr;
  QComboBox*                                odt_open_drt_creative_white_combo_   = nullptr;
  QWidget*                                  odt_open_drt_detail_panel_           = nullptr;

  std::vector<OpenDrtDetailSliderBinding>   odt_open_drt_detail_sliders_{};
  std::map<QSlider*, std::function<void()>> slider_reset_callbacks_{};
};

}  // namespace alcedo::ui
