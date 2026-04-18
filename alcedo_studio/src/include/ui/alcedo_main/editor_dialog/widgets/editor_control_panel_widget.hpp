//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

namespace alcedo::ui {

class EditorControlPanelWidget final : public QWidget {
 public:
  explicit EditorControlPanelWidget(QWidget* parent = nullptr);
};

}  // namespace alcedo::ui
