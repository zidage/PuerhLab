//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

/// @file album_backend_image_delete_test.cpp
/// @brief Image delete integration tests for AlbumBackend.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>

#include <duckdb.h>

#include <format>

namespace alcedo::ui::test {
namespace {

using DeleteTests = AlbumBackendTestFixture;

auto CountRows(const std::filesystem::path& dbPath, const std::string& sql) -> int64_t {
  duckdb_database   db = nullptr;
  duckdb_connection conn = nullptr;
  if (duckdb_open(dbPath.string().c_str(), &db) != DuckDBSuccess) {
    return -1;
  }
  if (duckdb_connect(db, &conn) != DuckDBSuccess) {
    duckdb_close(&db);
    return -1;
  }

  int64_t       count = -1;
  duckdb_result result;
  if (duckdb_query(conn, sql.c_str(), &result) == DuckDBSuccess) {
    count = duckdb_row_count(&result) > 0 ? duckdb_value_int64(&result, 0, 0) : 0;
  }
  duckdb_destroy_result(&result);
  duckdb_disconnect(&conn);
  duckdb_close(&db);
  return count;
}

void WaitForImportFinished(AlbumBackend& backend, int timeoutMs = 30000) {
  QSignalSpy spy(&backend, &AlbumBackend::ImportStateChanged);
  const int  step = 200;
  int        elapsed = 0;
  while (backend.ImportRunning() && elapsed < timeoutMs) {
    spy.wait(step);
    elapsed += step;
  }
  ProcessEvents(600);
}

auto CreateDirectProject(AlbumBackend& backend, const std::filesystem::path& dbPath,
                         const std::filesystem::path& metaPath) -> bool {
  {
    ProjectService project(dbPath, metaPath, ProjectOpenMode::kCreateNew);
    project.SaveProject(metaPath);
  }

  QSignalSpy projectSpy(&backend, &AlbumBackend::ProjectChanged);
  if (!backend.LoadProject(PathToQString(metaPath))) {
    return false;
  }
  WaitForSignal(projectSpy, 15000);
  ProcessEvents(500);
  return backend.ServiceReady();
}

TEST_F(DeleteTests, DeleteImages_RemovesImageAndRelatedRows) {
  sl_element_id_t deletedElementId = 0;
  image_id_t      deletedImageId   = 0;
  QVariantMap     result;
  int             shownCountAfterDelete = -1;

  {
    AlbumBackend backend;
    ASSERT_TRUE(CreateDirectProject(backend, db_path_, meta_path_));

    auto images = CollectRawTestImages("airplane", 1);
    if (images.empty()) {
      GTEST_SKIP() << "No RAW images available in raw/airplane/";
    }

    backend.StartImport(PathsToQStringList(images));
    WaitForImportFinished(backend);
    ASSERT_FALSE(backend.Thumbnails().isEmpty());

    const QVariantMap first = backend.Thumbnails().front().toMap();
    deletedElementId = static_cast<sl_element_id_t>(first.value("elementId").toUInt());
    deletedImageId   = static_cast<image_id_t>(first.value("imageId").toUInt());
    ASSERT_NE(deletedElementId, 0u);
    ASSERT_NE(deletedImageId, 0u);

    // Touch editor once so pipeline/history entries are materialized in normal service flow.
    backend.OpenEditor(static_cast<uint>(deletedElementId), static_cast<uint>(deletedImageId));
    ProcessEvents(400);

    QVariantList targets;
    targets.push_back(QVariantMap{{"elementId", static_cast<uint>(deletedElementId)},
                                  {"imageId", static_cast<uint>(deletedImageId)}});
    result = backend.DeleteImages(targets);
    ProcessEvents(500);

    shownCountAfterDelete = backend.ShownCount();
  }

  EXPECT_TRUE(result.value("success").toBool());
  EXPECT_EQ(result.value("deletedCount").toInt(), 1);
  EXPECT_EQ(result.value("failedCount").toInt(), 0);
  EXPECT_EQ(shownCountAfterDelete, 0);

  EXPECT_EQ(CountRows(db_path_, std::format("SELECT COUNT(*) FROM Image WHERE id={}", deletedImageId)),
            0);
  EXPECT_EQ(CountRows(db_path_,
                      std::format("SELECT COUNT(*) FROM FileImage WHERE file_id={}",
                                  deletedElementId)),
            0);
  EXPECT_EQ(CountRows(db_path_,
                      std::format("SELECT COUNT(*) FROM EditHistory WHERE file_id={}",
                                  deletedElementId)),
            0);
  EXPECT_EQ(CountRows(db_path_,
                      std::format("SELECT COUNT(*) FROM PipelineParam WHERE file_id={}",
                                  deletedElementId)),
            0);
}

TEST_F(DeleteTests, DeleteImages_BestEffortPartialFailure) {
  image_id_t  deletedImageId = 0;
  QVariantMap result;

  {
    AlbumBackend backend;
    ASSERT_TRUE(CreateDirectProject(backend, db_path_, meta_path_));

    auto images = CollectRawTestImages("airplane", 1);
    if (images.empty()) {
      GTEST_SKIP() << "No RAW images available in raw/airplane/";
    }

    backend.StartImport(PathsToQStringList(images));
    WaitForImportFinished(backend);
    ASSERT_FALSE(backend.Thumbnails().isEmpty());

    const QVariantMap first = backend.Thumbnails().front().toMap();
    const auto validElementId = static_cast<sl_element_id_t>(first.value("elementId").toUInt());
    deletedImageId = static_cast<image_id_t>(first.value("imageId").toUInt());

    QVariantList targets;
    targets.push_back(QVariantMap{{"elementId", static_cast<uint>(validElementId)},
                                  {"imageId", static_cast<uint>(deletedImageId)}});
    targets.push_back(QVariantMap{{"elementId", 999999u}, {"imageId", 999999u}});

    result = backend.DeleteImages(targets);
    ProcessEvents(500);
  }

  EXPECT_TRUE(result.value("success").toBool());
  EXPECT_EQ(result.value("deletedCount").toInt(), 1);
  EXPECT_EQ(result.value("failedCount").toInt(), 1);
  EXPECT_EQ(CountRows(db_path_, std::format("SELECT COUNT(*) FROM Image WHERE id={}", deletedImageId)),
            0);
}

}  // namespace
}  // namespace alcedo::ui::test
