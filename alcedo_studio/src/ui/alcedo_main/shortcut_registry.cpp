//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/shortcut_registry.hpp"

#include <QAction>
#include <QObject>

namespace alcedo::ui {

ShortcutRegistry::ShortcutRegistry(QWidget* owner) : owner_(owner) {}

auto ShortcutRegistry::Register(ShortcutBindingSpec spec) -> QAction* {
  if (!owner_ || spec.id.isEmpty() || !spec.on_trigger) {
    return nullptr;
  }

  if (const auto it = entries_.find(spec.id); it != entries_.end()) {
    return it->second.action;
  }

  auto* action = new QAction(owner_);
  action->setObjectName(spec.id);
  action->setShortcut(spec.default_sequence);
  action->setAutoRepeat(true);
  action->setShortcutContext(spec.context);
  if (!spec.description.isEmpty()) {
    action->setText(spec.description);
    action->setToolTip(spec.description);
    action->setStatusTip(spec.description);
  }
  owner_->addAction(action);

  const ShortcutCommandId id = spec.id;
  entries_.emplace(id, Entry{.spec = std::move(spec), .action = action});
  QObject::connect(action, &QAction::triggered, owner_, [this, id](bool) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) {
      return;
    }

    auto& entry = it->second;
    if (entry.spec.enabled_when) {
      const bool enabled = entry.spec.enabled_when();
      entry.action->setEnabled(enabled);
      if (!enabled) {
        return;
      }
    }

    entry.spec.on_trigger();
  });

  RefreshEnabledStates();
  return action;
}

auto ShortcutRegistry::Action(const ShortcutCommandId& id) const -> QAction* {
  if (const auto it = entries_.find(id); it != entries_.end()) {
    return it->second.action;
  }
  return nullptr;
}

auto ShortcutRegistry::ShortcutText(const ShortcutCommandId&     id,
                                    QKeySequence::SequenceFormat format) const -> QString {
  const auto* action = Action(id);
  if (!action) {
    return {};
  }
  return action->shortcut().toString(format);
}

auto ShortcutRegistry::DecorateTooltip(const QString&           base_tooltip,
                                       const ShortcutCommandId& id) const -> QString {
  const QString shortcut_text = ShortcutText(id);
  if (shortcut_text.isEmpty()) {
    return base_tooltip;
  }
  if (base_tooltip.isEmpty()) {
    return shortcut_text;
  }
  return QStringLiteral("%1 (%2)").arg(base_tooltip, shortcut_text);
}

void ShortcutRegistry::RefreshEnabledStates() {
  for (auto& [id, entry] : entries_) {
    Q_UNUSED(id);
    const bool enabled = !entry.spec.enabled_when || entry.spec.enabled_when();
    entry.action->setEnabled(enabled);
  }
}

}  // namespace alcedo::ui
