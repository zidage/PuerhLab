//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// GUI session tests. ApplicationModuleHost runs with the headless sink in this process, opens a
// CI RAW file in the editor, and serves a `gui` automation session: agent commands, the user
// adapters, and the notifications of a watching socket client meet in one editor session.

#include <gtest/gtest.h>

#include <QCoreApplication>
#include <QDeadlineTimer>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonObject>
#include <QLocalSocket>
#include <QTemporaryDir>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/editor_session_types.hpp"
#include "automation/automation_protocol.hpp"
#include "ui/album_backend_test_fixture.hpp"
#include "ui/alcedo_main/album_backend/agent_control_owner.hpp"
#include "ui/alcedo_main/automation/automation_host_session.hpp"
#include "ui/alcedo_main/automation/headless_frame_sink.hpp"
#include "ui/alcedo_main/automation/headless_host.hpp"

namespace alcedo::automation {
namespace {

using ui::ApplicationModuleHost;

auto CollectCiRawFiles() -> std::vector<std::filesystem::path> {
  const std::filesystem::path        root{std::string(TEST_IMG_PATH) + "/ci_rawfiles"};
  std::vector<std::filesystem::path> paths;
  if (std::filesystem::exists(root)) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.is_regular_file() && is_supported_file(entry.path())) {
        paths.push_back(entry.path());
      }
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

template <class Predicate>
auto WaitUntil(Predicate&& predicate, std::chrono::milliseconds timeout) -> bool {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return predicate();
}

/// The first editor backend of this platform and build.
auto PlatformEditorBackend() -> std::optional<editor_rhi::EditorBackend> {
  for (const auto backend : {editor_rhi::EditorBackend::OpenCl, editor_rhi::EditorBackend::Metal,
                             editor_rhi::EditorBackend::Cuda}) {
    if (editor_rhi::IsBackendSupportedOnThisPlatform(backend) &&
        editor_rhi::IsBackendAvailableInThisBuild(backend)) {
      return backend;
    }
  }
  return std::nullopt;
}

/// A socket client of the session that collects the notifications.
class WatchingClient {
 public:
  auto Connect(const QString& socket_name) -> bool {
    socket_.connectToServer(socket_name);
    return WaitUntil([this]() { return socket_.state() == QLocalSocket::ConnectedState; },
                     std::chrono::seconds(5));
  }

  /// Waits until a notification with @p method arrives and returns its params.
  auto WaitForNotification(const QString& method, std::chrono::milliseconds timeout)
      -> std::optional<QJsonObject> {
    std::optional<QJsonObject> found;
    WaitUntil(
        [&]() {
          buffer_.append(socket_.readAll());
          qsizetype newline = 0;
          while ((newline = buffer_.indexOf('\n')) >= 0) {
            const AutomationParseResult parsed = ParseAutomationLine(buffer_.left(newline));
            buffer_.remove(0, newline + 1);
            if (!parsed.message.has_value()) {
              continue;
            }
            if (const auto* notification = std::get_if<AutomationNotification>(&*parsed.message);
                notification != nullptr && notification->method == method && !found.has_value()) {
              found = notification->params;
            }
          }
          return found.has_value();
        },
        timeout);
    return found;
  }

  /// Drops the messages that arrived so far.
  void Drain() {
    QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
    buffer_.append(socket_.readAll());
    buffer_.clear();
  }

 private:
  QLocalSocket socket_;
  QByteArray   buffer_;
};

class AutomationGuiSessionTest : public ui::test::ApplicationModuleHostTestFixture {
 protected:
  void SetUp() override {
    ApplicationModuleHostTestFixture::SetUp();
    const auto backend = PlatformEditorBackend();
    if (!backend.has_value()) {
      GTEST_SKIP() << "no editor backend is available on this platform";
    }
    const auto raw_files = CollectCiRawFiles();
    ASSERT_FALSE(raw_files.empty()) << "the CI RAW files are missing under " << TEST_IMG_PATH;
    const QString backend_error = StartHeadlessEditorBackend(*backend);
    ASSERT_TRUE(backend_error.isEmpty()) << backend_error.toStdString();

    sink_ = std::make_unique<HeadlessFrameSink>();
    host_ = std::make_unique<ApplicationModuleHost>();
    host_->project()->SetRuntimeAcceleratorPreference(HeadlessAcceleratorPreference(*backend));
    ASSERT_TRUE(host_->editor_session()->BindHeadlessPresentationSink(sink_.get(), 1280, 720));
    ASSERT_TRUE(CreateTestProject(*host_));

    host_->import_export()->StartImport(ui::test::PathsToQStringList({raw_files.front()}));
    ASSERT_TRUE(WaitUntil([&] { return !host_->import_export()->ImportRunning(); },
                          std::chrono::minutes(2)));
    ASSERT_EQ(host_->import_export()->ImportFailed(), 0);
    ASSERT_TRUE(WaitUntil([&] { return !host_->library()->Thumbnails().isEmpty(); },
                          std::chrono::seconds(30)));
    const QVariantMap item     = host_->library()->Thumbnails().front().toMap();
    const uint        element  = item.value("elementId").toUInt();
    const uint        image_id = item.value("imageId").toUInt();
    ASSERT_NE(element, 0U);
    host_->workspace_router()->OpenEditor(element, image_id);
    ASSERT_TRUE(WaitUntil(
        [&] {
          return host_->editor_session_service()->state() == EditorSessionState::Interactive &&
                 host_->editor_session()->can_edit();
        },
        std::chrono::minutes(2)))
        << "state " << EditorSessionStateName(host_->editor_session_service()->state());

    ASSERT_TRUE(session_dir_.isValid());
    AutomationHostSessionOptions options;
    options.host_mode    = QStringLiteral("gui");
    options.session_name = QStringLiteral("gui");
    options.session_dir  = session_dir_.path();
    session_             = std::make_unique<AutomationHostSession>(host_.get(), std::move(options));
    QString error;
    ASSERT_TRUE(session_->Start(&error)) << error.toStdString();
  }

  void TearDown() override {
    if (session_) {
      session_->Stop();
      session_.reset();
    }
    if (host_) {
      host_->Shutdown();
      host_.reset();
    }
    sink_.reset();
    ApplicationModuleHostTestFixture::TearDown();
  }

  /// Dispatches one request through the session registry, the path of every socket request.
  auto Call(const QString& method, QJsonObject params = {}) -> std::optional<AutomationResponse> {
    auto              response = std::make_shared<std::optional<AutomationResponse>>();
    AutomationRequest request{QJsonValue(++next_id_), method, std::move(params)};
    session_->registry().Dispatch(request,
                                  AutomationReply(request.id, [response](AutomationResponse value) {
                                    *response = std::move(value);
                                  }));
    const QDeadlineTimer deadline(120000);
    while (!response->has_value() && !deadline.hasExpired()) {
      QCoreApplication::processEvents(QEventLoop::AllEvents | QEventLoop::WaitForMoreEvents, 50);
    }
    return *response;
  }

  auto Result(const QString& method, QJsonObject params = {}) -> QJsonObject {
    const auto response = Call(method, std::move(params));
    if (!response.has_value()) {
      ADD_FAILURE() << method.toStdString() << " did not answer";
      return {};
    }
    if (response->error.has_value()) {
      ADD_FAILURE() << method.toStdString()
                    << " failed: " << response->error->message.toStdString();
      return {};
    }
    return response->result.toObject();
  }

  auto ErrorCode(const QString& method, QJsonObject params = {}) -> std::optional<int> {
    const auto response = Call(method, std::move(params));
    if (!response.has_value() || !response->error.has_value()) {
      return std::nullopt;
    }
    return static_cast<int>(response->error->code);
  }

  auto SetExposure(double value) -> std::optional<AutomationResponse> {
    return Call(QStringLiteral("editor.set"), QJsonObject{{"field", "exposure"}, {"value", value}});
  }

  auto          control() -> ui::AgentControlOwner* { return host_->agent_control(); }

  QTemporaryDir session_dir_;
  std::unique_ptr<HeadlessFrameSink>     sink_;
  std::unique_ptr<ApplicationModuleHost> host_;
  std::unique_ptr<AutomationHostSession> session_;
  int                                    next_id_ = 0;
};

TEST_F(AutomationGuiSessionTest, AgentEditAppearsInControllerProjection) {
  const QJsonObject acquired =
      Result(QStringLiteral("control.acquire"), QJsonObject{{"holder_name", "test-agent"}});
  ASSERT_TRUE(acquired.value("held").toBool());
  EXPECT_EQ(acquired.value("holder_name").toString(), QStringLiteral("test-agent"));
  ASSERT_TRUE(control()->held());
  EXPECT_FALSE(host_->interaction_policy()->CanChangeStateAsUser());

  const QVariantMap snapshot_before = host_->editor_session()->adjustment_snapshot();
  const std::size_t commits_before  = host_->editor_session()->history_snapshot().commits.size();
  const auto        set             = SetExposure(1.25);
  ASSERT_TRUE(set.has_value());
  ASSERT_FALSE(set->error.has_value()) << set->error->message.toStdString();

  // The agent commit shows in the controller that QML reads.
  EXPECT_EQ(host_->editor_session()->history_snapshot().commits.size(), commits_before + 1);
  EXPECT_TRUE(host_->editor_session()->history_snapshot().can_undo);
  EXPECT_TRUE(
      WaitUntil([&] { return host_->editor_session()->adjustment_snapshot() != snapshot_before; },
                std::chrono::seconds(30)));
  EXPECT_NEAR(Result(QStringLiteral("editor.get"), QJsonObject{{"fields", QJsonArray{"exposure"}}})
                  .value("values")
                  .toObject()
                  .value("exposure")
                  .toDouble(),
              1.25, 1e-6);

  // The command ended: the agent task waits for the next command.
  const QVariantList tasks = host_->background_tasks()->Tasks();
  const auto         task  = std::find_if(tasks.begin(), tasks.end(), [&](const QVariant& value) {
    return value.toMap().value("id").toString() == control()->task_id();
  });
  ASSERT_NE(task, tasks.end());
  EXPECT_EQ(task->toMap().value("detail").toString(), ui::AgentControlOwner::WaitingDetail());

  // The user cannot write while the agent holds control.
  EXPECT_FALSE(host_->editor_session()->submitPatch(QStringLiteral("exposure"),
                                                    QStringLiteral("{\"exposure\":0.25}"), true));

  EXPECT_FALSE(Result(QStringLiteral("control.release")).value("held").toBool());
  EXPECT_TRUE(host_->interaction_policy()->CanChangeStateAsUser());
}

TEST_F(AutomationGuiSessionTest, StateChangeWithoutControlIsRejected) {
  EXPECT_FALSE(Result(QStringLiteral("control.get")).value("held").toBool());
  EXPECT_FALSE(
      Result(QStringLiteral("state.get")).value("control").toObject().value("held").toBool());
  const std::size_t commits_before = host_->editor_session()->history_snapshot().commits.size();
  EXPECT_EQ(
      ErrorCode(QStringLiteral("editor.set"), QJsonObject{{"field", "exposure"}, {"value", 1.0}}),
      static_cast<int>(AutomationErrorCode::ControlNotHeld));
  // A read runs without control.
  EXPECT_FALSE(Result(QStringLiteral("editor.history")).isEmpty());
  EXPECT_EQ(host_->editor_session()->history_snapshot().commits.size(), commits_before);

  // After the user takes control, the agent is refused until the user returns control.
  ASSERT_TRUE(Result(QStringLiteral("control.acquire")).value("held").toBool());
  ASSERT_TRUE(host_->background_tasks()->CancelTask(control()->task_id()));
  EXPECT_TRUE(host_->interaction_policy()->CanChangeStateAsUser() ||
              WaitUntil([&] { return host_->interaction_policy()->CanChangeStateAsUser(); },
                        std::chrono::seconds(5)));
  EXPECT_EQ(
      ErrorCode(QStringLiteral("editor.set"), QJsonObject{{"field", "exposure"}, {"value", 1.0}}),
      static_cast<int>(AutomationErrorCode::ControlRevokedByUser));
  EXPECT_EQ(ErrorCode(QStringLiteral("control.acquire")),
            static_cast<int>(AutomationErrorCode::ControlRevokedByUser));
  EXPECT_TRUE(Result(QStringLiteral("control.get")).value("revoked_by_user").toBool());
  EXPECT_EQ(host_->editor_session()->history_snapshot().commits.size(), commits_before);

  control()->ReturnControl();
  ASSERT_TRUE(Result(QStringLiteral("control.acquire")).value("held").toBool());
  const auto set = SetExposure(1.0);
  ASSERT_TRUE(set.has_value());
  EXPECT_FALSE(set->error.has_value());
  EXPECT_EQ(host_->editor_session()->history_snapshot().commits.size(), commits_before + 1);
}

TEST_F(AutomationGuiSessionTest, UserEditSendsHistoryChangedToWatchingClient) {
  WatchingClient client;
  ASSERT_TRUE(client.Connect(session_->socket_name()));
  client.Drain();

  // The user edits in the GUI without an agent: the watching client hears of the commit.
  ASSERT_TRUE(host_->editor_session()->submitPatch(QStringLiteral("exposure"),
                                                   QStringLiteral("{\"exposure\":0.75}"), true));
  const auto history =
      client.WaitForNotification(QStringLiteral("history.changed"), std::chrono::seconds(60));
  ASSERT_TRUE(history.has_value()) << "no history.changed for the user edit";

  // Agent control changes reach the client too.
  ASSERT_TRUE(Result(QStringLiteral("control.acquire")).value("held").toBool());
  const auto changed =
      client.WaitForNotification(QStringLiteral("control.changed"), std::chrono::seconds(5));
  ASSERT_TRUE(changed.has_value());
  EXPECT_TRUE(changed->value("held").toBool());
}

}  // namespace
}  // namespace alcedo::automation
