//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <map>

#include <QKeySequence>
#include <QString>
#include <QWidget>

class QAction;

namespace puerhlab::ui {

using ShortcutCommandId = QString;

struct ShortcutBindingSpec {
  ShortcutCommandId    id;
  QString              description;
  QKeySequence         default_sequence;
  Qt::ShortcutContext  context = Qt::WidgetWithChildrenShortcut;
  std::function<bool()> enabled_when{};
  std::function<void()> on_trigger{};
};

class ShortcutRegistry final {
 public:
  explicit ShortcutRegistry(QWidget* owner);

  auto Register(ShortcutBindingSpec spec) -> QAction*;
  auto Action(const ShortcutCommandId& id) const -> QAction*;
  auto ShortcutText(const ShortcutCommandId& id,
                    QKeySequence::SequenceFormat format = QKeySequence::NativeText) const
      -> QString;
  auto DecorateTooltip(const QString& base_tooltip, const ShortcutCommandId& id) const
      -> QString;
  void RefreshEnabledStates();

 private:
  struct Entry {
    ShortcutBindingSpec spec;
    QAction*            action = nullptr;
  };

  QWidget*                     owner_ = nullptr;
  std::map<ShortcutCommandId, Entry> entries_{};
};

}  // namespace puerhlab::ui
