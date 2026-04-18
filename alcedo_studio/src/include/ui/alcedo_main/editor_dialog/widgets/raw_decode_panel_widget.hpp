//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QWidget>

namespace alcedo::ui {

class RawDecodePanelWidget final : public QWidget {
 public:
  explicit RawDecodePanelWidget(QWidget* parent = nullptr);
};

}  // namespace alcedo::ui
