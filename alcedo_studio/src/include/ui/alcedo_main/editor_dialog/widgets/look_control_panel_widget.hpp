//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "ui/alcedo_main/editor_dialog/controllers/lut_controller.hpp"
#include "ui/alcedo_main/editor_dialog/session/adjustment_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/look_adjustment_state.hpp"

namespace alcedo::ui {

class CdlTrackballDiscWidget;
class EditorAdjustmentSession;
class LutBrowserWidget;

class LookControlPanelWidget final : public AdjustmentPanelWidget {
 public:
  struct Dependencies {
    EditorAdjustmentSession* session                = nullptr;
    AdjustmentState*         dialog_state           = nullptr;
    AdjustmentState*         dialog_committed_state = nullptr;
  };

  struct Callbacks {
    std::function<bool()>                                is_global_syncing;
    std::function<void()>                                request_render;
    std::function<void(QSlider*, std::function<void()>)> register_slider_reset;
    std::function<const AdjustmentState&()>              default_adjustment_state;
    std::function<std::optional<LookAdjustmentState>(const LookAdjustmentState&)>
        load_from_pipeline;
  };

  explicit LookControlPanelWidget(QWidget* parent = nullptr);

  void Configure(Dependencies deps, Callbacks callbacks);
  void Build();

  auto PanelId() const -> AdjustmentPanelId override { return AdjustmentPanelId::Look; }
  void LoadFromPipeline() override;
  void ReloadFromCommittedState() override;
  void SetSyncing(bool syncing) override;

  void SyncControlsFromDialogState();
  void RetranslateUi();
  void RefreshLutBrowserUi(bool force_refresh = false, bool preserve_scroll_position = false);
  auto DefaultLutPath() -> std::string;
  auto SelectRelativeLut(int step) -> bool;
  auto CanHandleLutNavigationShortcut(QWidget* focus_widget) const -> bool;

 private:
  void BuildLutSection();
  void BuildHlsSection();
  void BuildCdlSection();

  void ProjectLookStateToDialog();
  void PullLookStateFromDialog();
  void PullCommittedLookStateFromDialog();
  void CopyLookStateToDialogState(const LookAdjustmentState& look_state, AdjustmentState& state);

  auto IsSyncing() const -> bool;
  void RequestPipelineRender();
  void PreviewLookField(AdjustmentField field);
  void CommitLookField(AdjustmentField field);
  void ResetHlsField(
      const std::function<void(LookAdjustmentState&, const AdjustmentState&)>& apply_default);

  auto ActiveHlsProfileIndex() const -> int;
  void SaveActiveHlsProfile();
  void LoadActiveHlsProfile();
  void RefreshHlsTargetUi();
  void RefreshCdlOffsetLabels();
  void SyncCdlControlsFromState();

  Dependencies deps_{};
  Callbacks    callbacks_{};
  bool         local_syncing_ = false;
  bool         built_         = false;

  LookAdjustmentState look_state_{};
  LookAdjustmentState committed_look_state_{};

  QVBoxLayout*             layout_                       = nullptr;
  LutBrowserWidget*        lut_browser_widget_           = nullptr;
  QLabel*                  hls_target_label_             = nullptr;
  std::vector<QPushButton*> hls_candidate_buttons_{};
  QSlider*                 hls_hue_adjust_slider_        = nullptr;
  QSlider*                 hls_lightness_adjust_slider_  = nullptr;
  QSlider*                 hls_saturation_adjust_slider_ = nullptr;
  QSlider*                 hls_hue_range_slider_         = nullptr;
  CdlTrackballDiscWidget*  lift_disc_widget_             = nullptr;
  CdlTrackballDiscWidget*  gamma_disc_widget_            = nullptr;
  CdlTrackballDiscWidget*  gain_disc_widget_             = nullptr;
  QLabel*                  lift_offset_label_            = nullptr;
  QLabel*                  gamma_offset_label_           = nullptr;
  QLabel*                  gain_offset_label_            = nullptr;
  QSlider*                 lift_master_slider_           = nullptr;
  QSlider*                 gamma_master_slider_          = nullptr;
  QSlider*                 gain_master_slider_           = nullptr;

  controllers::LutController lut_controller_{};
};

}  // namespace alcedo::ui
