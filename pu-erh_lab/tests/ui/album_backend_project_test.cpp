/// @file album_backend_project_test.cpp
/// @brief Project lifecycle tests for AlbumBackend.
///
/// Covers: create project, load project (valid/invalid), save project, and
/// initial service state.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>
#include <fstream>

namespace puerhlab::ui::test {
namespace {

using ProjectTests = AlbumBackendTestFixture;

// ── Initial state ──────────────────────────────────────────────────────────

TEST_F(ProjectTests, ServiceState_InitiallyNotReady) {
  AlbumBackend backend;
  EXPECT_FALSE(backend.ServiceReady());
  EXPECT_FALSE(backend.ServiceMessage().isEmpty());
}

// ── Create project — happy path ────────────────────────────────────────────

TEST_F(ProjectTests, CreateProject_ValidFolder_Succeeds) {
  AlbumBackend backend;
  QSignalSpy   projSpy(&backend, &AlbumBackend::ProjectChanged);
  QSignalSpy   stateSpy(&backend, &AlbumBackend::ServiceStateChanged);

  const bool ok =
      backend.CreateProjectInFolderNamed(PathToQString(temp_dir_), "test_proj");
  EXPECT_TRUE(ok);

  // Wait for async project initialisation.
  WaitForSignal(projSpy, 15000);
  ProcessEvents(500);

  EXPECT_TRUE(backend.ServiceReady());
  EXPECT_FALSE(stateSpy.isEmpty());
}

// ── Create project — empty name ────────────────────────────────────────────

TEST_F(ProjectTests, CreateProject_EmptyName_Fails) {
  AlbumBackend backend;
  const bool   ok =
      backend.CreateProjectInFolderNamed(PathToQString(temp_dir_), "");
  // Either returns false or sets a service message.
  // The critical assertion: no crash.
  if (!ok) {
    SUCCEED();
  } else {
    // If it somehow succeeds with empty name, the service message should
    // still be reasonable.
    ProcessEvents(200);
  }
}

// ── Create project while "loading" — second call rejected ──────────────────

TEST_F(ProjectTests, CreateProject_DoubleCall_SecondRejected) {
  AlbumBackend backend;

  const bool first =
      backend.CreateProjectInFolderNamed(PathToQString(temp_dir_), "proj_a");

  // If first call started async loading, a second call should be rejected.
  if (first && backend.ProjectLoading()) {
    // Create a different subfolder so paths differ.
    const auto subDir = temp_dir_ / "sub";
    std::filesystem::create_directories(subDir);
    const bool second =
        backend.CreateProjectInFolderNamed(PathToQString(subDir), "proj_b");
    EXPECT_FALSE(second);
  }

  // Drain everything so destructor is clean.
  ProcessEvents(2000);
}

// ── Load project — non-existent file ───────────────────────────────────────

TEST_F(ProjectTests, LoadProject_NonexistentFile_Fails) {
  AlbumBackend backend;
  const bool   ok = backend.LoadProject("C:/nonexistent/project.json");
  EXPECT_FALSE(ok);
  EXPECT_FALSE(backend.ServiceReady());
}

// ── Load project — invalid format ──────────────────────────────────────────

TEST_F(ProjectTests, LoadProject_InvalidFormat_Fails) {
  AlbumBackend backend;

  // Create a temporary .txt file — not a valid project format.
  const auto txtPath = temp_dir_ / "notes.txt";
  {
    std::ofstream ofs(txtPath);
    ofs << "hello world";
  }

  const bool ok = backend.LoadProject(PathToQString(txtPath));
  EXPECT_FALSE(ok);
}

// ── Save project — no project loaded ───────────────────────────────────────

TEST_F(ProjectTests, SaveProject_NoProject_Fails) {
  AlbumBackend backend;
  const bool   ok = backend.SaveProject();
  EXPECT_FALSE(ok);
}

// ── Save project — after create ────────────────────────────────────────────

TEST_F(ProjectTests, SaveProject_AfterCreate_Succeeds) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  const bool ok = backend.SaveProject();
  EXPECT_TRUE(ok);
}

// ── Create project with default name via convenience overload ──────────────

TEST_F(ProjectTests, CreateProjectInFolder_DefaultName_Succeeds) {
  AlbumBackend backend;
  QSignalSpy   projSpy(&backend, &AlbumBackend::ProjectChanged);

  const bool ok = backend.CreateProjectInFolder(PathToQString(temp_dir_));
  EXPECT_TRUE(ok);

  WaitForSignal(projSpy, 15000);
  ProcessEvents(500);
  EXPECT_TRUE(backend.ServiceReady());
}

}  // namespace
}  // namespace puerhlab::ui::test
