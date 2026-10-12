//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QObject>
#include <QString>
#include <memory>
#include <optional>
#include <vector>

#include "app/project_service.hpp"
#include "automation/automation_protocol.hpp"
#include "storage/store/sleeve/element_store.hpp"
#include "ui/alcedo_main/album_backend/application_module_host.hpp"
#include "ui/alcedo_main/automation/automation_command_registry.hpp"

namespace alcedo::automation {

/// `{"type": "object", "properties": ..., "required": ...}`. Omits `required` when empty.
auto AutomationObjectSchema(QJsonObject properties = {}, QJsonArray required = {}) -> QJsonObject;

/// A parameter schema: an object schema with `"additionalProperties": false`.
auto AutomationClosedParamsSchema(QJsonObject properties = {}, QJsonArray required = {})
    -> QJsonObject;

/// The project service of the entered project, or null. Sends -32001 when no project is open,
/// a project is loading, or a project is loaded but not entered (the window shows the start
/// screen). The message names the next command; `data.path` is the loaded project file, if any.
auto AutomationEnteredProject(ui::ApplicationModuleHost* host, AutomationReply& reply)
    -> std::shared_ptr<ProjectService>;

/// The library rows of @p params["element_ids"], in that order, without repeated ids. Sends
/// -32602 with `data.unknown_ids` and returns nothing when an id is not a photo of the project.
auto ResolveAutomationElementRows(ProjectService& project, const QJsonObject& params,
                                  AutomationReply& reply)
    -> std::optional<std::vector<SearchResultRow>>;

/// Sends @p code with @p message and `data.reason` set to @p message, the form of `rejected`,
/// `failed`, and `not_ready` errors that report an owner message.
void SendAutomationOwnerError(AutomationReply& reply, AutomationErrorCode code,
                              const QString& message);

/// Sends -32602 with `data.pointer` set to @p pointer (a JSON pointer into the parameters) and
/// `data.reason` set to @p message.
void SendAutomationParamError(AutomationReply& reply, const QString& pointer,
                              const QString& message);

/// The reply of one command that waits for owner signals. Connect the owner signals with this
/// object as the context: the first Send call answers the request and deletes this object with
/// deleteLater, which ends those connections. GUI thread only.
class AutomationCommandWait final : public QObject {
 public:
  explicit AutomationCommandWait(AutomationReply reply, QObject* parent = nullptr);

  void SendResult(QJsonValue result);
  void SendError(AutomationErrorCode code, const QString& message,
                 QJsonValue data = QJsonValue(QJsonValue::Undefined));
  void SendOwnerError(AutomationErrorCode code, const QString& message);
  /// True after a Send call or after the registry sent the timeout response of the request.
  [[nodiscard]] auto IsAnswered() const -> bool { return finished_ || reply_.IsSent(); }

 private:
  void            Finish();

  AutomationReply reply_;
  bool            finished_ = false;
};

}  // namespace alcedo::automation
