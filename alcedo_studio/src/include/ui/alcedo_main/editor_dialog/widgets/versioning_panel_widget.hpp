//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

namespace alcedo::ui {

class VersioningPanelWidget final : public QWidget {
 public:
  explicit VersioningPanelWidget(QWidget* parent = nullptr);
};

}  // namespace alcedo::ui
