//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/automation/automation_command_support.hpp"

#include <QStringList>
#include <algorithm>
#include <exception>
#include <utility>

namespace alcedo::automation {

auto AutomationObjectSchema(QJsonObject properties, QJsonArray required) -> QJsonObject {
  QJsonObject schema{{"type", "object"}, {"properties", properties}};
  if (!required.isEmpty()) {
    schema.insert("required", required);
  }
  return schema;
}

auto AutomationClosedParamsSchema(QJsonObject properties, QJsonArray required) -> QJsonObject {
  QJsonObject schema = AutomationObjectSchema(std::move(properties), std::move(required));
  schema.insert("additionalProperties", false);
  return schema;
}

auto AutomationEnteredProject(ui::ApplicationModuleHost* host, AutomationReply& reply)
    -> std::shared_ptr<ProjectService> {
  ui::ProjectModule* project = host->project();
  auto               service = project->handler().project();
  if (service && project->ProjectEntered() && !project->ProjectLoading()) {
    return service;
  }
  const QString path = QString::fromStdWString(project->handler().package_path().generic_wstring());
  QString       message;
  if (project->ProjectLoading()) {
    message = QStringLiteral(
        "A project is loading; wait until state.get reports project.loading false.");
  } else if (service) {
    // The GUI can start with the last project loaded while the window stays on the start screen.
    message = QStringLiteral(
        "The project is loaded but not open in the window; call project.open with its path.");
  } else {
    message = QStringLiteral("No project is open; call project.open or project.create.");
  }
  QJsonObject data{{"reason", message}};
  if (service) {
    data.insert("path", path);
  }
  reply.SendError(AutomationErrorCode::NotReady, message, data);
  return nullptr;
}

auto ResolveAutomationElementRows(ProjectService& project, const QJsonObject& params,
                                  AutomationReply& reply)
    -> std::optional<std::vector<SearchResultRow>> {
  std::vector<sl_element_id_t> ids;
  for (const QJsonValue& value : params.value("element_ids").toArray()) {
    const auto id = static_cast<sl_element_id_t>(value.toDouble());
    if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
      ids.push_back(id);
    }
  }
  const auto browse = project.GetAlbumBrowseService();
  if (!browse) {
    SendAutomationOwnerError(reply, AutomationErrorCode::Failed,
                             QStringLiteral("The library cannot be read."));
    return std::nullopt;
  }
  std::vector<SearchResultRow> rows;
  try {
    rows = browse->ReadAlbumFileRows(ids);
  } catch (const std::exception& e) {
    SendAutomationOwnerError(reply, AutomationErrorCode::Failed, QString::fromUtf8(e.what()));
    return std::nullopt;
  }
  QJsonArray  unknown;
  QStringList unknown_text;
  for (const auto id : ids) {
    const bool found = std::any_of(rows.begin(), rows.end(), [id](const SearchResultRow& row) {
      return row.file_id_ == id && row.image_id_ != 0;
    });
    if (!found) {
      unknown.push_back(static_cast<double>(id));
      unknown_text.push_back(QString::number(id));
    }
  }
  if (!unknown.isEmpty()) {
    reply.SendError(
        AutomationErrorCode::InvalidParams,
        QStringLiteral("unknown element ids: %1").arg(unknown_text.join(QStringLiteral(", "))),
        QJsonObject{{"pointer", "/element_ids"}, {"unknown_ids", unknown}});
    return std::nullopt;
  }
  return rows;
}

void SendAutomationOwnerError(AutomationReply& reply, AutomationErrorCode code,
                              const QString& message) {
  reply.SendError(code, message, QJsonObject{{"reason", message}});
}

void SendAutomationParamError(AutomationReply& reply, const QString& pointer,
                              const QString& message) {
  reply.SendError(AutomationErrorCode::InvalidParams, message,
                  QJsonObject{{"pointer", pointer}, {"reason", message}});
}

AutomationCommandWait::AutomationCommandWait(AutomationReply reply, QObject* parent)
    : QObject(parent), reply_(std::move(reply)) {}

void AutomationCommandWait::SendResult(QJsonValue result) {
  if (finished_) {
    return;
  }
  reply_.SendResult(std::move(result));
  Finish();
}

void AutomationCommandWait::SendError(AutomationErrorCode code, const QString& message,
                                      QJsonValue data) {
  if (finished_) {
    return;
  }
  reply_.SendError(code, message, std::move(data));
  Finish();
}

void AutomationCommandWait::SendOwnerError(AutomationErrorCode code, const QString& message) {
  if (finished_) {
    return;
  }
  SendAutomationOwnerError(reply_, code, message);
  Finish();
}

void AutomationCommandWait::Finish() {
  finished_ = true;
  deleteLater();
}

}  // namespace alcedo::automation
