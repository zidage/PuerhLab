//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

namespace alcedo::ui {

enum class AdjustmentPanelId {
  Tone,
  Look,
  DisplayTransform,
  Geometry,
  RawDecode,
};

class AdjustmentPanelWidget : public QWidget {
 public:
  using QWidget::QWidget;
  ~AdjustmentPanelWidget() override                 = default;

  virtual auto PanelId() const -> AdjustmentPanelId = 0;
  virtual void LoadFromPipeline()                   = 0;
  virtual void ReloadFromCommittedState()           = 0;
  virtual void SetSyncing(bool syncing)             = 0;
};

}  // namespace alcedo::ui
