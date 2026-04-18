//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>

#include "ui/alcedo_main/i18n.hpp"

namespace alcedo::ui::test {
namespace {

using AlbumBackendI18nTests = AlbumBackendTestFixture;

TEST_F(AlbumBackendI18nTests, InitialLocalizedStrings_AreAvailable) {
  AlbumBackend backend;

  EXPECT_FALSE(backend.ServiceMessage().isEmpty());
  EXPECT_FALSE(backend.TaskStatus().isEmpty());
  EXPECT_FALSE(backend.ExportStatus().isEmpty());
  EXPECT_FALSE(backend.EditorStatus().isEmpty());
}

TEST_F(AlbumBackendI18nTests, TranslationNotifier_RefreshesObservableStateSignals) {
  AlbumBackend backend;

  QSignalSpy service_spy(&backend, &AlbumBackend::ServiceStateChanged);
  QSignalSpy task_spy(&backend, &AlbumBackend::TaskStateChanged);
  QSignalSpy import_spy(&backend, &AlbumBackend::ImportStateChanged);
  QSignalSpy export_spy(&backend, &AlbumBackend::ExportStateChanged);
  QSignalSpy editor_spy(&backend, &AlbumBackend::EditorStateChanged);
  QSignalSpy project_spy(&backend, &AlbumBackend::ProjectLoadStateChanged);

  i18n::TranslationNotifier::Instance().NotifyLanguageChanged();
  ProcessEvents(100);

  EXPECT_GE(service_spy.count(), 1);
  EXPECT_GE(task_spy.count(), 1);
  EXPECT_GE(import_spy.count(), 1);
  EXPECT_GE(export_spy.count(), 1);
  EXPECT_GE(editor_spy.count(), 1);
  EXPECT_GE(project_spy.count(), 1);
}

}  // namespace
}  // namespace alcedo::ui::test
