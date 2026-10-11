//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/agent_control_owner.hpp"

#include <QMetaObject>
#include <QPointer>
#include <algorithm>
#include <utility>

#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui {

AgentControlOwner::AgentControlOwner(BackgroundTaskController* tasks, QObject* parent)
    : QObject(parent), tasks_(tasks) {}

void AgentControlOwner::SetUnsealedInputReader(UnsealedInputReader reader) {
  unsealed_input_reader_ = std::move(reader);
}

auto AgentControlOwner::HasUnsealedInput() const -> bool {
  return unsealed_input_reader_ && unsealed_input_reader_();
}

auto AgentControlOwner::TryAcquire(const QString& holder_name) -> AgentControlRefusal {
  if (revoked_by_user_) {
    return AgentControlRefusal::RevokedByUser;
  }
  if (held_) {
    return holder_name == holder_name_ ? AgentControlRefusal::None
                                       : AgentControlRefusal::HeldByOtherAgent;
  }
  if (HasUnsealedInput()) {
    return AgentControlRefusal::UnsealedEditorInput;
  }

  BackgroundTaskSnapshot snapshot;
  snapshot.kind_             = BackgroundTaskKind::AgentControl;
  snapshot.state_            = BackgroundTaskState::Running;
  snapshot.title_            = holder_name;
  snapshot.detail_           = WaitingDetail();
  snapshot.progress_percent_ = -1;
  snapshot.cancelable_       = true;
  snapshot.shutdown_policy_  = BackgroundTaskShutdownPolicy::CancelAndWait;
  snapshot.locks_.push_back(InteractionLock{InteractionCapability::UserStateChange, 0,
                                            Tr("An agent controls the session.")});
  held_        = true;
  holder_name_ = holder_name;
  running_commands_.clear();

  // The task registry calls the cancel callback inside CancelTask, before CancelTask returns.
  // RevokeByUser changes the control state at once and finishes the task after CancelTask.
  QPointer<AgentControlOwner> self = this;
  task_id_                         = tasks_->RegisterTask(snapshot, [self]() {
    if (self != nullptr) {
      self->RevokeByUser();
    }
  });
  emit ControlChanged();
  return AgentControlRefusal::None;
}

void AgentControlOwner::RequestAcquire(const QString& holder_name, AcquireWaitCheck still_waiting,
                                       AcquireDone done) {
  const AgentControlRefusal refusal = TryAcquire(holder_name);
  if (refusal != AgentControlRefusal::UnsealedEditorInput) {
    done(refusal);
    return;
  }
  pending_acquires_.push_back(
      PendingAcquire{holder_name, std::move(still_waiting), std::move(done)});
}

void AgentControlOwner::RecheckPendingAcquires() {
  // A done callback can start another request, so the list is taken first.
  std::vector<PendingAcquire> pending = std::move(pending_acquires_);
  pending_acquires_.clear();
  for (auto& request : pending) {
    if (request.still_waiting_ && !request.still_waiting_()) {
      continue;
    }
    const AgentControlRefusal refusal = TryAcquire(request.holder_name_);
    if (refusal == AgentControlRefusal::UnsealedEditorInput) {
      pending_acquires_.push_back(std::move(request));
      continue;
    }
    request.done_(refusal);
  }
}

auto AgentControlOwner::Release() -> bool {
  if (!held_) {
    return false;
  }
  held_ = false;
  holder_name_.clear();
  running_commands_.clear();
  tasks_->FinishTask(task_id_, BackgroundTaskState::Succeeded, Tr("Control returned to you."));
  emit ControlChanged();
  return true;
}

void AgentControlOwner::TakeControl() {
  if (held_) {
    (void)tasks_->CancelTask(task_id_);
  }
}

void AgentControlOwner::RevokeByUser() {
  if (!held_) {
    return;
  }
  held_            = false;
  revoked_by_user_ = true;
  holder_name_.clear();
  running_commands_.clear();
  // FinishTask drops the cancel callback that is running now, so it runs after CancelTask.
  const QString task_id = task_id_;
  QMetaObject::invokeMethod(
      this,
      [this, task_id]() {
        tasks_->FinishTask(task_id, BackgroundTaskState::Canceled, Tr("You took control."));
      },
      Qt::QueuedConnection);
  emit ControlChanged();
}

void AgentControlOwner::ReturnControl() {
  if (!revoked_by_user_) {
    return;
  }
  revoked_by_user_ = false;
  emit ControlChanged();
}

auto AgentControlOwner::BeginCommand(const QString& summary) -> quint64 {
  if (!held_) {
    return 0;
  }
  const quint64 token = ++next_command_token_;
  running_commands_.push_back(RunningCommand{token, summary});
  PublishTaskDetail();
  return token;
}

void AgentControlOwner::EndCommand(quint64 token) {
  const auto found =
      std::find_if(running_commands_.begin(), running_commands_.end(),
                   [token](const RunningCommand& command) { return command.token_ == token; });
  if (found == running_commands_.end()) {
    return;
  }
  running_commands_.erase(found);
  PublishTaskDetail();
}

auto AgentControlOwner::WaitingDetail() -> QString { return Tr("Waiting for the next command"); }

void AgentControlOwner::PublishTaskDetail() {
  if (!held_) {
    return;
  }
  const QString detail =
      running_commands_.empty() ? WaitingDetail() : running_commands_.back().summary_;
  tasks_->UpdateTask(task_id_, holder_name_, detail, -1);
}

}  // namespace alcedo::ui
