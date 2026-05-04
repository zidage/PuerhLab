//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <functional>
#include <map>
#include <optional>

#include "ui/alcedo_main/editor_dialog/modules/geometry.hpp"
#include "ui/alcedo_main/editor_dialog/session/adjustment_panel.hpp"
#include "ui/alcedo_main/editor_dialog/state.hpp"
#include "ui/alcedo_main/editor_dialog/state/geometry_adjustment_state.hpp"

namespace alcedo::ui {

class EditorAdjustmentSession;

class GeometryPanelWidget final : public AdjustmentPanelWidget {
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
    std::function<void(bool locked, float ratio)>           set_crop_overlay_aspect_lock;
    std::function<void(float x, float y, float w, float h)> set_crop_overlay_rect;
    std::function<void(float degrees)>                      set_crop_overlay_rotation;
    std::function<void(bool visible)>                       set_crop_overlay_visible;
    std::function<void(bool enabled)>                       set_crop_tool_enabled;
    std::function<float()>                                  source_aspect_ratio;
  };

  explicit GeometryPanelWidget(QWidget* parent = nullptr);

  void Configure(Dependencies deps, Callbacks callbacks);
  void Build();

  auto PanelId() const -> AdjustmentPanelId override { return AdjustmentPanelId::Geometry; }
  void LoadFromPipeline() override;
  void ReloadFromCommittedState() override;
  void SetSyncing(bool syncing) override;

  void SyncControlsFromDialogState();
  void RetranslateUi();

  auto ResetButton() const -> QPushButton* { return geometry_reset_btn_; }

  // Called from the shell when the viewer overlay changes.
  void SetCropRectFromViewer(float x, float y, float w, float h);
  void SetRotationFromViewer(float degrees);

  void ResetCropAndRotation();

 private:
  void         BuildCropAspectSection();
  void         BuildRotateSection();
  void         BuildCropOffsetSection();
  void         BuildApplyResetSection();

  void         ProjectGeometryStateToDialog();
  void         PullGeometryStateFromDialog();
  void         PullCommittedGeometryStateFromDialog();

  auto         IsSyncing() const -> bool;
  bool         eventFilter(QObject* obj, QEvent* event) override;
  void         RegisterSliderReset(QSlider* slider, std::function<void()> on_reset);
  void         RequestPipelineRender();

  void         PreviewGeometryField(AdjustmentField field);
  void         CommitGeometryField(AdjustmentField field);

  void         UpdateGeometryCropRectLabel();
  auto         CurrentGeometrySourceAspect() const -> float;
  auto         CurrentGeometryAspectRatio() const -> std::optional<float>;
  void         SyncGeometryCropSlidersFromState();
  void         SyncCropAspectControlsFromState();
  void         PushGeometryStateToOverlay();
  void         SetCropRectState(float x, float y, float w, float h, bool sync_controls = true,
                                bool sync_overlay = true);
  void         ApplyAspectPresetToCurrentCrop();
  void         ResizeCropRectWithAspect(float proposed_value, bool use_width_driver);
  void         SetCropAspectPresetState(geometry::CropAspectPreset preset);
  void         RefreshGeometryModeUi();

  Dependencies deps_{};
  Callbacks    callbacks_{};
  bool         local_syncing_ = false;

  GeometryAdjustmentState                   geometry_state_{};
  GeometryAdjustmentState                   committed_geometry_state_{};

  QSlider*                                  rotate_slider_                     = nullptr;
  QSlider*                                  geometry_crop_x_slider_            = nullptr;
  QSlider*                                  geometry_crop_y_slider_            = nullptr;
  QSlider*                                  geometry_crop_w_slider_            = nullptr;
  QSlider*                                  geometry_crop_h_slider_            = nullptr;
  QComboBox*                                geometry_crop_aspect_preset_combo_ = nullptr;
  QDoubleSpinBox*                           geometry_crop_aspect_width_spin_   = nullptr;
  QDoubleSpinBox*                           geometry_crop_aspect_height_spin_  = nullptr;
  QLabel*                                   geometry_crop_rect_label_          = nullptr;
  QPushButton*                              geometry_apply_btn_                = nullptr;
  QPushButton*                              geometry_reset_btn_                = nullptr;
  std::map<QSlider*, std::function<void()>> slider_reset_callbacks_{};
};

}  // namespace alcedo::ui
