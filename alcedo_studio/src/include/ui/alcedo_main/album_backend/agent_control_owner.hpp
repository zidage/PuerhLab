//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QObject>
#include <QString>
#include <QtGlobal>
#include <cstdint>
#include <functional>
#include <vector>

#include "ui/alcedo_main/album_backend/background_task_controller.hpp"

namespace alcedo::ui {

/// The reason why `AgentControlOwner` did not give control to an agent.
enum class AgentControlRefusal : std::uint8_t {
  None,
  /// The user took control back. The agent must wait until the user selects Return control.
  RevokedByUser,
  /// An agent with another holder name holds control.
  HeldByOtherAgent,
  /// The editor has user slider input that a Release or Cancel boundary has not sealed yet.
  UnsealedEditorInput,
};

/// Owner of agent control in the GUI session: the turn-taking state between the user and an
/// automation agent (CLI or MCP).
///
/// While an agent holds control, one running background task of the kind `AgentControl` holds
/// the global `UserStateChange` lock. The user-facing adapters read that lock and refuse user
/// state changes. The editor session admission does not read it, so the commands of the agent
/// still run. The existing task island and task dialog show the task. Canceling the task is
/// **Take control**: control goes back to the user at once, and the agent cannot acquire
/// control again until the user selects **Return control**.
///
/// Only this owner changes the control state (`held`, `holder_name`, `revoked_by_user`), and it
/// is the only caller of `UpdateTask` and `FinishTask` for its task. GUI thread only.
class AgentControlOwner final : public QObject {
  Q_OBJECT
  Q_PROPERTY(bool held READ held NOTIFY ControlChanged)
  Q_PROPERTY(bool revokedByUser READ revoked_by_user NOTIFY ControlChanged)
  Q_PROPERTY(QString holderName READ holder_name NOTIFY ControlChanged)
  /// The id of the newest agent task, also after the task finished. Empty before the first
  /// acquire.
  Q_PROPERTY(QString taskId READ task_id NOTIFY ControlChanged)

 public:
  /// Returns true while the editor has user input that no Release or Cancel boundary sealed.
  using UnsealedInputReader = std::function<bool()>;
  /// Returns true while the caller of RequestAcquire still waits for the answer.
  using AcquireWaitCheck    = std::function<bool()>;
  using AcquireDone         = std::function<void(AgentControlRefusal)>;

  /// @p tasks must outlive this owner.
  explicit AgentControlOwner(BackgroundTaskController* tasks, QObject* parent = nullptr);

  /// Sets the reader of the editor pending input. Without a reader the editor has no unsealed
  /// input.
  void               SetUnsealedInputReader(UnsealedInputReader reader);

  [[nodiscard]] auto held() const -> bool { return held_; }
  [[nodiscard]] auto revoked_by_user() const -> bool { return revoked_by_user_; }
  [[nodiscard]] auto holder_name() const -> QString { return holder_name_; }
  [[nodiscard]] auto task_id() const -> QString { return task_id_; }

  /// Gives control to @p holder_name and registers the agent task. Holding control already
  /// with the same holder name is a success without a change. Refuses without a change when the
  /// user revoked control, another holder holds it, or the editor has unsealed input.
  auto               TryAcquire(const QString& holder_name) -> AgentControlRefusal;
  /// TryAcquire, but when the editor has unsealed input, waits: RecheckPendingAcquires tries
  /// again until the acquire succeeds or is refused for another reason, or until @p still_waiting
  /// returns false (the caller stopped waiting, for example after its timeout). Calls @p done
  /// once with the outcome, never after @p still_waiting returned false.
  void RequestAcquire(const QString& holder_name, AcquireWaitCheck still_waiting, AcquireDone done);
  /// Tries the waiting acquires again. Call it when the editor pending input can have changed.
  void RecheckPendingAcquires();
  [[nodiscard]] auto pending_acquire_count() const -> std::size_t {
    return pending_acquires_.size();
  }

  /// The agent gives control back: the agent task finishes as Succeeded. Returns false and
  /// changes nothing when no agent holds control.
  auto             Release() -> bool;
  /// Take control: cancels the agent task. The task finishes as Canceled and the user has
  /// control at once. No change when no agent holds control.
  Q_INVOKABLE void TakeControl();
  /// Return control: the agent can acquire control again. It does not acquire control for the
  /// agent. No change when the user did not revoke control.
  Q_INVOKABLE void ReturnControl();

  /// A state-changing agent command starts: the agent task detail shows @p summary. Returns a
  /// token for EndCommand, or 0 when no agent holds control.
  auto             BeginCommand(const QString& summary) -> quint64;
  /// The command of @p token ended. The task detail shows the newest command that still runs,
  /// or the waiting text.
  void             EndCommand(quint64 token);
  /// The task detail between commands.
  static auto      WaitingDetail() -> QString;

 signals:
  /// `held`, `holder_name`, or `revoked_by_user` changed. Emitted once per change.
  void ControlChanged();

 private:
  struct PendingAcquire {
    QString          holder_name_;
    AcquireWaitCheck still_waiting_;
    AcquireDone      done_;
  };
  struct RunningCommand {
    quint64 token_ = 0;
    QString summary_;
  };

  void                        RevokeByUser();
  void                        PublishTaskDetail();
  [[nodiscard]] auto          HasUnsealedInput() const -> bool;

  BackgroundTaskController*   tasks_ = nullptr;
  UnsealedInputReader         unsealed_input_reader_;
  bool                        held_            = false;
  bool                        revoked_by_user_ = false;
  QString                     holder_name_;
  QString                     task_id_;
  std::vector<PendingAcquire> pending_acquires_;
  std::vector<RunningCommand> running_commands_;
  quint64                     next_command_token_ = 0;
};

}  // namespace alcedo::ui
