//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/agent_control_owner.hpp"

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QSignalSpy>
#include <QString>
#include <QVariantList>
#include <QVariantMap>
#include <optional>
#include <vector>

#include "ui/alcedo_main/album_backend/background_task_controller.hpp"
#include "ui/alcedo_main/album_backend/interaction_policy_controller.hpp"

namespace alcedo::ui {
namespace {

auto FindTask(const BackgroundTaskController& tasks, const QString& id) -> QVariantMap {
  for (const QVariant& value : tasks.Tasks()) {
    const QVariantMap task = value.toMap();
    if (task.value(QStringLiteral("id")).toString() == id) {
      return task;
    }
  }
  return {};
}

/// Runs the queued calls, for example the FinishTask that follows Take control.
void DrainEvents() {
  for (int i = 0; i < 4; ++i) {
    QCoreApplication::processEvents();
  }
}

TEST(AgentControlOwnerTest, AcquireWaitsForPendingInputRelease) {
  BackgroundTaskController tasks;
  AgentControlOwner        control(&tasks);
  bool                     unsealed = true;
  control.SetUnsealedInputReader([&unsealed]() { return unsealed; });

  EXPECT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::UnsealedEditorInput);
  EXPECT_FALSE(control.held());
  EXPECT_TRUE(tasks.Tasks().isEmpty());

  std::vector<AgentControlRefusal> outcomes;
  control.RequestAcquire(
      QStringLiteral("agent"), []() { return true; },
      [&outcomes](AgentControlRefusal refusal) { outcomes.push_back(refusal); });
  EXPECT_TRUE(outcomes.empty());
  EXPECT_EQ(control.pending_acquire_count(), 1u);

  // The user still drags the slider: the acquire keeps waiting.
  control.RecheckPendingAcquires();
  EXPECT_TRUE(outcomes.empty());
  EXPECT_FALSE(control.held());

  // The release boundary sealed the input: the acquire succeeds once.
  unsealed = false;
  control.RecheckPendingAcquires();
  ASSERT_EQ(outcomes.size(), 1u);
  EXPECT_EQ(outcomes.front(), AgentControlRefusal::None);
  EXPECT_TRUE(control.held());
  EXPECT_EQ(control.pending_acquire_count(), 0u);
  control.RecheckPendingAcquires();
  EXPECT_EQ(outcomes.size(), 1u);
}

TEST(AgentControlOwnerTest, AcquireWaitIsDroppedWhenTheCallerStopsWaiting) {
  BackgroundTaskController tasks;
  AgentControlOwner        control(&tasks);
  bool                     unsealed = true;
  control.SetUnsealedInputReader([&unsealed]() { return unsealed; });

  bool waiting = true;
  int  done    = 0;
  control.RequestAcquire(
      QStringLiteral("agent"), [&waiting]() { return waiting; },
      [&done](AgentControlRefusal) { ++done; });
  waiting  = false;
  unsealed = false;
  control.RecheckPendingAcquires();
  EXPECT_EQ(done, 0);
  EXPECT_FALSE(control.held());
  EXPECT_EQ(control.pending_acquire_count(), 0u);
}

TEST(AgentControlOwnerTest, AcquireRegistersRunningAgentTaskWithUserStateChangeLock) {
  BackgroundTaskController    tasks;
  InteractionPolicyController policy(&tasks);
  AgentControlOwner           control(&tasks);
  QSignalSpy                  changed(&control, &AgentControlOwner::ControlChanged);
  ASSERT_TRUE(policy.CanChangeStateAsUser());

  ASSERT_EQ(control.TryAcquire(QStringLiteral("claude-code")), AgentControlRefusal::None);
  EXPECT_TRUE(control.held());
  EXPECT_EQ(control.holder_name(), QStringLiteral("claude-code"));
  EXPECT_EQ(changed.count(), 1);

  const QVariantMap task = FindTask(tasks, control.task_id());
  ASSERT_FALSE(task.isEmpty());
  EXPECT_EQ(task.value(QStringLiteral("kind")).toString(), QStringLiteral("agentControl"));
  EXPECT_EQ(task.value(QStringLiteral("state")).toString(), QStringLiteral("running"));
  EXPECT_EQ(task.value(QStringLiteral("title")).toString(), QStringLiteral("claude-code"));
  EXPECT_EQ(task.value(QStringLiteral("detail")).toString(), AgentControlOwner::WaitingDetail());
  EXPECT_EQ(task.value(QStringLiteral("progressPercent")).toInt(), -1);
  EXPECT_TRUE(task.value(QStringLiteral("cancelable")).toBool());
  const QVariantList locks = task.value(QStringLiteral("locks")).toList();
  ASSERT_EQ(locks.size(), 1);
  EXPECT_EQ(locks.front().toMap().value(QStringLiteral("capability")).toString(),
            QStringLiteral("userStateChange"));

  EXPECT_FALSE(policy.CanChangeStateAsUser());
  EXPECT_FALSE(policy.ChangeStateAsUserReason().isEmpty());
  EXPECT_EQ(UserStateChangeRefusal(&policy), policy.ChangeStateAsUserReason());

  // The same holder acquires again without a second task; another holder is refused.
  EXPECT_EQ(control.TryAcquire(QStringLiteral("claude-code")), AgentControlRefusal::None);
  EXPECT_EQ(tasks.Tasks().size(), 1);
  EXPECT_EQ(control.TryAcquire(QStringLiteral("other")), AgentControlRefusal::HeldByOtherAgent);
  EXPECT_EQ(control.holder_name(), QStringLiteral("claude-code"));
  EXPECT_EQ(changed.count(), 1);
}

TEST(AgentControlOwnerTest, CommandSummaryUpdatesAgentTaskDetail) {
  BackgroundTaskController tasks;
  AgentControlOwner        control(&tasks);
  EXPECT_EQ(control.BeginCommand(QStringLiteral("editor.set exposure=0.5")), 0u);
  ASSERT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);

  const quint64 first = control.BeginCommand(QStringLiteral("editor.set exposure=0.5"));
  ASSERT_NE(first, 0u);
  EXPECT_EQ(FindTask(tasks, control.task_id()).value(QStringLiteral("detail")).toString(),
            QStringLiteral("editor.set exposure=0.5"));

  const quint64 second = control.BeginCommand(QStringLiteral("editor.undo"));
  EXPECT_EQ(FindTask(tasks, control.task_id()).value(QStringLiteral("detail")).toString(),
            QStringLiteral("editor.undo"));

  // The newest command that still runs shows; after the last one, the waiting text.
  control.EndCommand(second);
  EXPECT_EQ(FindTask(tasks, control.task_id()).value(QStringLiteral("detail")).toString(),
            QStringLiteral("editor.set exposure=0.5"));
  control.EndCommand(first);
  EXPECT_EQ(FindTask(tasks, control.task_id()).value(QStringLiteral("detail")).toString(),
            AgentControlOwner::WaitingDetail());
}

TEST(AgentControlOwnerTest, CancelingAgentTaskRevokesControl) {
  BackgroundTaskController    tasks;
  InteractionPolicyController policy(&tasks);
  AgentControlOwner           control(&tasks);
  ASSERT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);
  const QString task_id = control.task_id();

  // The task dialog cancels the task: Take control.
  ASSERT_TRUE(tasks.CancelTask(task_id));
  EXPECT_FALSE(control.held());
  EXPECT_TRUE(control.revoked_by_user());
  EXPECT_TRUE(control.holder_name().isEmpty());

  DrainEvents();
  const QVariantMap task = FindTask(tasks, task_id);
  EXPECT_EQ(task.value(QStringLiteral("state")).toString(), QStringLiteral("canceled"));
  EXPECT_TRUE(policy.CanChangeStateAsUser());
  EXPECT_EQ(control.task_id(), task_id);
  // A command that the agent started before the cancel does not write the detail again.
  EXPECT_EQ(control.BeginCommand(QStringLiteral("editor.set")), 0u);
}

TEST(AgentControlOwnerTest, ReacquireIsRefusedUntilReturnControl) {
  BackgroundTaskController tasks;
  AgentControlOwner        control(&tasks);
  ASSERT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);
  control.TakeControl();
  DrainEvents();
  ASSERT_TRUE(control.revoked_by_user());

  EXPECT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::RevokedByUser);
  EXPECT_EQ(control.TryAcquire(QStringLiteral("other")), AgentControlRefusal::RevokedByUser);
  EXPECT_FALSE(control.held());

  QSignalSpy changed(&control, &AgentControlOwner::ControlChanged);
  control.ReturnControl();
  EXPECT_FALSE(control.revoked_by_user());
  EXPECT_FALSE(control.held()) << "Return control does not acquire control for the agent";
  EXPECT_EQ(changed.count(), 1);

  ASSERT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);
  EXPECT_TRUE(control.held());
  EXPECT_EQ(tasks.Tasks().size(), 2);
}

TEST(AgentControlOwnerTest, ReleaseFinishesAgentTaskAsSucceeded) {
  BackgroundTaskController    tasks;
  InteractionPolicyController policy(&tasks);
  AgentControlOwner           control(&tasks);
  EXPECT_FALSE(control.Release());
  ASSERT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);
  const QString task_id = control.task_id();

  EXPECT_TRUE(control.Release());
  EXPECT_FALSE(control.held());
  EXPECT_FALSE(control.revoked_by_user());
  const QVariantMap task = FindTask(tasks, task_id);
  EXPECT_EQ(task.value(QStringLiteral("state")).toString(), QStringLiteral("succeeded"));
  EXPECT_TRUE(policy.CanChangeStateAsUser());
  EXPECT_EQ(tasks.ActiveCountExcept(BackgroundTaskKind::ImageAnalysis), 0);

  // A released agent can acquire again at once.
  EXPECT_EQ(control.TryAcquire(QStringLiteral("agent")), AgentControlRefusal::None);
}

}  // namespace
}  // namespace alcedo::ui
