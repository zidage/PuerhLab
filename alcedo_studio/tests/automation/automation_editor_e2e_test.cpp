//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// Editor commands and render.preview through real alcedo-cli and alcedo_main --headless
// processes on a CI RAW file. The CI preset runs these tests (label ci_automation_flow) with the
// default editor backend of the platform; on Windows a build with CUDA uses CUDA.

#include <gtest/gtest.h>

#include <QDeadlineTimer>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QStringList>
#include <algorithm>
#include <cmath>
#include <functional>

#include "automation/automation_client.hpp"
#include "automation/automation_session_file.hpp"
#include "automation_editor_e2e_session.hpp"

namespace alcedo::automation {
namespace {

using test::CliRun;

class AutomationEditorE2ETest : public test::AutomationEditorE2ETestBase {};

/// Reads notifications from @p watch until one has @p method and satisfies @p match, and returns
/// its params. Returns an empty object when none arrives within one minute.
auto WaitForNotification(AutomationClient& watch, const QString& method,
                         const std::function<bool(const QJsonObject&)>& match) -> QJsonObject {
  const QDeadlineTimer deadline(60000);
  while (!deadline.hasExpired()) {
    const auto notification = watch.ReadNotification(static_cast<int>(deadline.remainingTime()));
    if (!notification.has_value()) {
      break;
    }
    if (notification->method == method && match(notification->params)) {
      return notification->params;
    }
  }
  return {};
}

TEST_F(AutomationEditorE2ETest, SetExposureCreatesOneCommitAndReadsBackUiValue) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("set_project")).isEmpty());
  OpenPhoto();
  const qsizetype  commits_before = CommitCount();
  const QJsonValue head_before    = Head();
  // A value equal to the current one changes nothing and makes no commit.
  ASSERT_GT(std::abs(Value(QStringLiteral("exposure")) - 0.5), 0.1);
  ASSERT_GT(std::abs(Value(QStringLiteral("saturation")) - 20), 1);

  const QJsonObject set = Set(QStringLiteral("exposure"), 0.5);
  EXPECT_NEAR(set.value("value").toDouble(), 0.5, 1e-6);
  EXPECT_NE(set.value("head_commit"), head_before);
  EXPECT_EQ(CommitCount(), commits_before + 1);
  EXPECT_EQ(Head(), set.value("head_commit"));
  EXPECT_NEAR(Value(QStringLiteral("exposure")), 0.5, 1e-6);

  // The issue example: a percent field in UI units.
  EXPECT_NEAR(Set(QStringLiteral("saturation"), 20).value("value").toDouble(), 20, 1e-4);
  EXPECT_NEAR(Value(QStringLiteral("saturation")), 20, 1e-4);
  EXPECT_EQ(CommitCount(), commits_before + 2);

  const QJsonValue  head = Head();
  const QJsonObject same = Set(QStringLiteral("saturation"), 20);
  EXPECT_EQ(same.value("head_commit"), head);
  EXPECT_EQ(CommitCount(), commits_before + 2);
  Stop();
}

TEST_F(AutomationEditorE2ETest, UndoRestoresPreviousUiValue) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("undo_project")).isEmpty());
  OpenPhoto();
  const double before = Value(QStringLiteral("exposure"));
  ASSERT_GT(std::abs(before - 2.0), 0.1);
  const QJsonValue  head_before = Head();
  const QJsonValue  head_after  = Set(QStringLiteral("exposure"), 2.0).value("head_commit");

  const QJsonObject undone      = Result(QStringLiteral("editor.undo"));
  EXPECT_EQ(undone.value("head_commit"), head_before);
  EXPECT_TRUE(undone.value("can_redo").toBool());
  EXPECT_NEAR(Value(QStringLiteral("exposure")), before, 1e-6);

  const QJsonObject redone = Result(QStringLiteral("editor.redo"));
  EXPECT_EQ(redone.value("head_commit"), head_after);
  EXPECT_NEAR(Value(QStringLiteral("exposure")), 2.0, 1e-6);
  Stop();
}

TEST_F(AutomationEditorE2ETest, BatchStopsAtFirstInvalidValueAndReportsApplied) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("batch_project")).isEmpty());
  OpenPhoto();
  const qsizetype commits_before  = CommitCount();
  const double    contrast_before = Value(QStringLiteral("contrast"));
  ASSERT_GT(std::abs(Value(QStringLiteral("exposure")) - 0.5), 0.1);
  ASSERT_GT(std::abs(contrast_before - 10), 1);

  const CliRun run =
      Call(QStringLiteral("editor.batch_set"),
           QJsonObject{{"changes", QJsonArray{QJsonObject{{"field", "exposure"}, {"value", 0.5}},
                                              QJsonObject{{"field", "saturation"}, {"value", 1000}},
                                              QJsonObject{{"field", "contrast"}, {"value", 10}}}}});
  EXPECT_EQ(run.exit_code, 1);
  const QJsonObject error = run.json.value("error").toObject();
  EXPECT_EQ(error.value("code").toInt(), -32602);
  const QJsonObject data = error.value("data").toObject();
  EXPECT_EQ(data.value("applied").toInt(), 1);
  EXPECT_EQ(data.value("pointer").toString(), QStringLiteral("/changes/1/value"));
  EXPECT_EQ(data.value("head_commit"), Head());
  EXPECT_EQ(CommitCount(), commits_before + 1);
  EXPECT_NEAR(Value(QStringLiteral("exposure")), 0.5, 1e-6);
  EXPECT_NEAR(Value(QStringLiteral("contrast")), contrast_before, 1e-6)
      << "changes after the failure stay out";

  const QJsonObject applied = Result(
      QStringLiteral("editor.batch_set"),
      QJsonObject{{"changes", QJsonArray{QJsonObject{{"field", "contrast"}, {"value", 10}},
                                         QJsonObject{{"field", "vibrance"}, {"value", -5}}}}});
  EXPECT_EQ(applied.value("applied").toInt(), 2);
  EXPECT_EQ(CommitCount(), commits_before + 3);
  EXPECT_NEAR(applied.value("values").toObject().value("vibrance").toDouble(), -5, 1e-4);
  Stop();
}

TEST_F(AutomationEditorE2ETest, SetBeforeInteractiveIsRejectedWithTheOwnerMessage) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("rejected_project")).isEmpty());
  const CliRun run =
      Call(QStringLiteral("editor.set"), QJsonObject{{"field", "exposure"}, {"value", 1.0}});
  EXPECT_EQ(run.exit_code, 1);
  const QJsonObject error = run.json.value("error").toObject();
  EXPECT_EQ(error.value("code").toInt(), -32002);
  EXPECT_FALSE(error.value("message").toString().isEmpty());
  EXPECT_EQ(error.value("data").toObject().value("reason"), error.value("message"));
  Stop();
}

TEST_F(AutomationEditorE2ETest, PreviewWritesCurrentAndRootPngWithRequestedLongEdge) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("preview_project")).isEmpty());
  OpenPhoto();
  ASSERT_GT(std::abs(Value(QStringLiteral("exposure")) - 1.0), 0.1);
  Set(QStringLiteral("exposure"), 1.0);
  const QString     out_dir = QDir(root_).filePath(QStringLiteral("preview"));
  const QJsonObject preview = Result(QStringLiteral("render.preview"),
                                     QJsonObject{{"out_dir", out_dir}, {"long_edge", 512}});
  for (const char* key : {"current", "root"}) {
    const QJsonObject file = preview.value(QLatin1String(key)).toObject();
    const QString     path = file.value("path").toString();
    EXPECT_TRUE(path.endsWith(QStringLiteral("_%1.png").arg(QLatin1String(key)))) << key;
    EXPECT_EQ(QFileInfo(path).absolutePath(), QFileInfo(out_dir).absoluteFilePath()) << key;
    const QImage image(path);
    ASSERT_FALSE(image.isNull()) << path.toStdString();
    EXPECT_EQ(std::max(image.width(), image.height()), 512) << key;
    EXPECT_EQ(file.value("width").toInt(), image.width()) << key;
    EXPECT_EQ(file.value("height").toInt(), image.height()) << key;
  }
  const QImage current(preview.value("current").toObject().value("path").toString());
  const QImage root(preview.value("root").toObject().value("path").toString());
  EXPECT_EQ(current.size(), root.size());
  EXPECT_NE(current, root) << "the current image has the exposure edit";
  Stop();
}

TEST_F(AutomationEditorE2ETest, RegionPreviewHasRegionAspectRatio) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("region_project")).isEmpty());
  OpenPhoto();
  const QString     out_dir = QDir(root_).filePath(QStringLiteral("full"));
  const QJsonObject full    = Result(QStringLiteral("render.preview"),
                                     QJsonObject{{"out_dir", out_dir}, {"long_edge", 1024}})
                               .value("current")
                               .toObject();
  const double full_width  = full.value("width").toDouble();
  const double full_height = full.value("height").toDouble();
  ASSERT_GT(full_width, 0);
  ASSERT_GT(full_height, 0);

  const QJsonObject region{{"x", 0.1}, {"y", 0.2}, {"w", 0.5}, {"h", 0.25}};
  const QJsonObject part =
      Result(QStringLiteral("render.preview"),
             QJsonObject{{"out_dir", QDir(root_).filePath(QStringLiteral("region"))},
                         {"long_edge", 1024},
                         {"region", region}})
          .value("current")
          .toObject();
  const double width  = part.value("width").toDouble();
  const double height = part.value("height").toDouble();
  ASSERT_GT(width, 0);
  ASSERT_GT(height, 0);
  // The region is 0.5 by 0.25 of the image: height = width * (0.25 H) / (0.5 W), to a pixel.
  const double expected_height = width * (0.25 * full_height) / (0.5 * full_width);
  EXPECT_LE(std::abs(height - expected_height), 1.0)
      << width << "x" << height << " for a " << full_width << "x" << full_height << " image";

  const CliRun outside =
      Call(QStringLiteral("render.preview"),
           QJsonObject{{"out_dir", out_dir},
                       {"region", QJsonObject{{"x", 0.8}, {"y", 0.0}, {"w", 0.5}, {"h", 0.5}}}});
  EXPECT_EQ(outside.json.value("error").toObject().value("code").toInt(), -32602);
  Stop();
}

// Issue #355: compare 'none' writes only the current file, and file_stem names the files.
TEST_F(AutomationEditorE2ETest, PreviewWithoutCompareWritesOnlyCurrentFileWithFileStem) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("stem_project")).isEmpty());
  OpenPhoto();
  const QString     out_dir = QDir(root_).filePath(QStringLiteral("stem"));
  const QJsonObject preview = Result(
      QStringLiteral("render.preview"),
      QJsonObject{
          {"out_dir", out_dir}, {"long_edge", 256}, {"compare", "none"}, {"file_stem", "shot_7"}});
  EXPECT_FALSE(preview.contains("root"));
  const QString current_path = preview.value("current").toObject().value("path").toString();
  EXPECT_EQ(QFileInfo(current_path).fileName(), QStringLiteral("shot_7_current.png"));
  const QImage current(current_path);
  ASSERT_FALSE(current.isNull()) << current_path.toStdString();
  EXPECT_EQ(std::max(current.width(), current.height()), 256);
  EXPECT_EQ(QDir(out_dir).entryList(QDir::Files), QStringList{QStringLiteral("shot_7_current.png")})
      << "no root file is written";

  const QJsonObject both =
      Result(QStringLiteral("render.preview"),
             QJsonObject{{"out_dir", out_dir}, {"long_edge", 256}, {"file_stem", "shot_7"}});
  EXPECT_EQ(QFileInfo(both.value("root").toObject().value("path").toString()).fileName(),
            QStringLiteral("shot_7_root.png"))
      << "compare defaults to 'root'";

  const CliRun folder = Call(QStringLiteral("render.preview"),
                             QJsonObject{{"out_dir", out_dir}, {"file_stem", "../outside"}});
  EXPECT_EQ(folder.json.value("error").toObject().value("code").toInt(), -32602);
  Stop();
}

TEST_F(AutomationEditorE2ETest, CropRatioPresetGivesPresetAspectInPixels) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("crop_ratio_project")).isEmpty());
  OpenPhoto();
  const QJsonObject root =
      Result(QStringLiteral("render.preview"),
             QJsonObject{{"out_dir", QDir(root_).filePath(QStringLiteral("root"))},
                         {"long_edge", 1400}})
          .value("root")
          .toObject();
  const double source_aspect = root.value("width").toDouble() / root.value("height").toDouble();
  // A square source would hide a constraint in normalized coordinates.
  ASSERT_GT(std::abs(source_aspect - 1.0), 0.1) << "the CI RAW file is square";

  // The write of the issue: the 2.35:1 preset is a pixel ratio, not 1 / 2.35 of the height.
  Result(QStringLiteral("editor.set"),
         QJsonObject{{"field", "crop_rotate"},
                     {"value", QJsonObject{{"x", 0.0},
                                           {"y", 0.31},
                                           {"width", 1.0},
                                           {"height", 0.638},
                                           {"angle_degrees", 0.0},
                                           {"aspect_preset", "ratio_2_35_1_35mm"}}}});
  const QJsonObject crop =
      Result(QStringLiteral("editor.get"), QJsonObject{{"fields", QJsonArray{"crop_rotate"}}})
          .value("values")
          .toObject()
          .value("crop_rotate")
          .toObject();
  EXPECT_EQ(crop.value("aspect_preset").toString(), QStringLiteral("ratio_2_35_1_35mm"));
  // source_aspect comes from a preview rounded to whole pixels.
  EXPECT_NEAR(crop.value("width").toDouble() * source_aspect / crop.value("height").toDouble(),
              2.35, 5e-3);

  const QJsonObject current =
      Result(QStringLiteral("render.preview"),
             QJsonObject{{"out_dir", QDir(root_).filePath(QStringLiteral("cropped"))},
                         {"long_edge", 1400}})
          .value("current")
          .toObject();
  const double width  = current.value("width").toDouble();
  const double height = current.value("height").toDouble();
  ASSERT_GT(width, 0);
  ASSERT_GT(height, 0);
  EXPECT_LE(std::abs(height - width / 2.35), 1.0) << width << "x" << height;
  Stop();
}

TEST_F(AutomationEditorE2ETest, EditsPersistAfterSessionRestart) {
  const QString project_path = StartWithOnePhoto(QStringLiteral("persist_project"));
  ASSERT_FALSE(project_path.isEmpty());
  OpenPhoto();
  ASSERT_GT(std::abs(Value(QStringLiteral("exposure")) - 0.75), 0.1);
  ASSERT_GT(std::abs(Value(QStringLiteral("saturation")) - 15), 1);
  Set(QStringLiteral("exposure"), 0.75);
  Set(QStringLiteral("saturation"), 15);
  const QJsonObject history = Result(QStringLiteral("editor.history"));
  EXPECT_TRUE(Result(QStringLiteral("editor.close")).value("closed").toBool());
  Stop();

  ASSERT_TRUE(StartWithProject(project_path));
  OpenPhoto();
  EXPECT_NEAR(Value(QStringLiteral("exposure")), 0.75, 1e-6);
  EXPECT_NEAR(Value(QStringLiteral("saturation")), 15, 1e-4);
  const QJsonObject reopened = Result(QStringLiteral("editor.history"));
  EXPECT_EQ(reopened.value("head_commit"), history.value("head_commit"));
  EXPECT_EQ(reopened.value("commits").toArray().size(), history.value("commits").toArray().size());
  Stop();
}

// Issue #356: editor and history changes reach a client that watches the session.
TEST_F(AutomationEditorE2ETest, WatchReceivesSessionStateAndHistoryNotifications) {
  ASSERT_FALSE(StartWithOnePhoto(QStringLiteral("notify_project")).isEmpty());
  const auto session_file =
      ReadAutomationSessionFile(AutomationSessionFilePath(session_dir_, QStringLiteral("e2e")));
  ASSERT_TRUE(session_file.has_value());
  // The server takes this connection before the connections of the commands below, so the
  // watch receives every notification that they cause.
  AutomationClient watch;
  ASSERT_TRUE(watch.Connect(session_file->socket_name, 10000))
      << watch.error_string().toStdString();
  const double      element = element_.toDouble();

  const QJsonObject opened  = OpenPhoto();
  const QJsonObject state   = WaitForNotification(
      watch, QStringLiteral("session.state_changed"), [element](const QJsonObject& params) {
        return params.value("state").toString() == QStringLiteral("Interactive") &&
               params.value("element_id").toDouble() == element;
      });
  ASSERT_FALSE(state.isEmpty()) << "no session.state_changed for editor.open";
  EXPECT_EQ(state.value("image_id").toDouble(), opened.value("image_id").toDouble());

  const QJsonValue  head    = Set(QStringLiteral("exposure"), 0.5).value("head_commit");
  const QJsonObject history = WaitForNotification(
      watch, QStringLiteral("history.changed"),
      [head](const QJsonObject& params) { return params.value("head_commit") == head; });
  ASSERT_FALSE(history.isEmpty()) << "no history.changed for editor.set";
  EXPECT_EQ(history.value("element_id").toDouble(), element);
  EXPECT_EQ(history.value("image_id").toDouble(), opened.value("image_id").toDouble());
  const QJsonValue first_version = history.value("version_id");
  EXPECT_FALSE(first_version.toString().isEmpty());

  const QJsonValue version_id =
      Result(QStringLiteral("versions.create"), QJsonObject{{"name", "Notified"}})
          .value("version_id");
  ASSERT_NE(version_id, first_version);
  const QJsonObject version = WaitForNotification(
      watch, QStringLiteral("history.changed"),
      [version_id](const QJsonObject& params) { return params.value("version_id") == version_id; });
  EXPECT_FALSE(version.isEmpty()) << "no history.changed for versions.create";

  Result(QStringLiteral("editor.close"));
  const QJsonObject closed = WaitForNotification(
      watch, QStringLiteral("session.state_changed"), [](const QJsonObject& params) {
        return params.value("state").toString() != QStringLiteral("Interactive");
      });
  EXPECT_FALSE(closed.isEmpty()) << "no session.state_changed for editor.close";
  Stop();
}

}  // namespace
}  // namespace alcedo::automation
