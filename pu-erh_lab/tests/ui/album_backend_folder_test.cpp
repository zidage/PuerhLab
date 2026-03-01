/// @file album_backend_folder_test.cpp
/// @brief Folder CRUD & selection tests for AlbumBackend.
///
/// Covers: folder creation, deletion, selection (valid/invalid IDs), and
/// signal emission.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>

namespace puerhlab::ui::test {
namespace {

using FolderTests = AlbumBackendTestFixture;

// ── Create folder — signal emitted, folder visible ─────────────────────────

TEST_F(FolderTests, CreateFolder_ValidName_EmitsFoldersChanged) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  QSignalSpy foldersChangedSpy(&backend, &AlbumBackend::FoldersChanged);

  backend.CreateFolder("TestSubFolder");
  ProcessEvents(500);

  EXPECT_FALSE(foldersChangedSpy.isEmpty())
      << "FoldersChanged should fire after CreateFolder";

  // Verify the folder appears in the folder list.
  const QVariantList folders = backend.Folders();
  bool found = false;
  for (const auto& v : folders) {
    const auto map = v.toMap();
    if (map.value("name").toString() == "TestSubFolder") {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Created folder should appear in Folders() list. "
                     << "Got " << folders.size() << " folders.";
}

// ── Select folder — valid ID ───────────────────────────────────────────────

TEST_F(FolderTests, SelectFolder_ValidId_EmitsFolderSelectionChanged) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Create a subfolder first.
  backend.CreateFolder("SubFolderA");
  ProcessEvents(500);

  const QVariantList folders = backend.Folders();
  ASSERT_FALSE(folders.isEmpty());

  // Find the subfolder ID.
  uint subFolderId = 0;
  for (const auto& v : folders) {
    const auto map = v.toMap();
    if (map.value("name").toString() == "SubFolderA") {
      subFolderId = map.value("folderId").toUInt();
      break;
    }
  }
  ASSERT_NE(subFolderId, 0u) << "SubFolderA must have a valid ID";

  QSignalSpy selSpy(&backend, &AlbumBackend::FolderSelectionChanged);
  backend.SelectFolder(subFolderId);
  ProcessEvents(200);

  EXPECT_FALSE(selSpy.isEmpty());
  EXPECT_EQ(backend.CurrentFolderId(), subFolderId);
}

// ── Select folder — invalid ID ─────────────────────────────────────────────

TEST_F(FolderTests, SelectFolder_InvalidId_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Select a folder ID that cannot exist.
  backend.SelectFolder(999999);
  ProcessEvents(200);

  // No crash is the assertion.
  SUCCEED();
}

// ── Delete folder ──────────────────────────────────────────────────────────

TEST_F(FolderTests, DeleteFolder_ExistingId_Removes) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  backend.CreateFolder("ToBeDeleted");
  ProcessEvents(500);

  const QVariantList before = backend.Folders();
  uint               targetId = 0;
  for (const auto& v : before) {
    const auto map = v.toMap();
    if (map.value("name").toString() == "ToBeDeleted") {
      targetId = map.value("folderId").toUInt();
      break;
    }
  }
  ASSERT_NE(targetId, 0u);

  QSignalSpy foldersChangedSpy(&backend, &AlbumBackend::FoldersChanged);
  backend.DeleteFolder(targetId);
  ProcessEvents(500);

  EXPECT_FALSE(foldersChangedSpy.isEmpty());

  const QVariantList after = backend.Folders();
  for (const auto& v : after) {
    const auto map = v.toMap();
    EXPECT_NE(map.value("folderId").toUInt(), targetId)
        << "Deleted folder should no longer appear";
  }
}

// ── Delete folder — invalid ID ─────────────────────────────────────────────

TEST_F(FolderTests, DeleteFolder_InvalidId_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  backend.DeleteFolder(0);
  backend.DeleteFolder(888888);
  ProcessEvents(200);

  SUCCEED();
}

// ── Folder operations without project — no crash ───────────────────────────

TEST_F(FolderTests, FolderOps_NoProject_NoCrash) {
  AlbumBackend backend;
  // No project created — all folder ops should be no-ops.

  backend.CreateFolder("ShouldNotWork");
  backend.SelectFolder(1);
  backend.DeleteFolder(1);
  ProcessEvents(200);

  EXPECT_FALSE(backend.ServiceReady());
}

}  // namespace
}  // namespace puerhlab::ui::test
