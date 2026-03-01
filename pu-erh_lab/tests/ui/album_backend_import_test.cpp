/// @file album_backend_import_test.cpp
/// @brief Import-robustness tests for AlbumBackend.
///
/// Focus: JPEG/TIFF graceful handling (pipeline does not support them as raw
/// decode input), CTD prevention, mixed-format import, cancellation, and
/// edge-case inputs.  All tests run headlessly via QCoreApplication.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>
#include <filesystem>

namespace puerhlab::ui::test {
namespace {

using ImportTests = AlbumBackendTestFixture;

// ── Helper: wait until importRunning becomes false ─────────────────────────

void WaitForImportFinished(AlbumBackend& backend, int timeoutMs = 30000) {
  QSignalSpy spy(&backend, &AlbumBackend::ImportStateChanged);
  const int step = 200;
  int       elapsed = 0;
  while (backend.ImportRunning() && elapsed < timeoutMs) {
    spy.wait(step);
    elapsed += step;
  }
  // Drain remaining queued events (FinishImport is posted via
  // Qt::QueuedConnection).
  ProcessEvents(500);
}

// ── Single RAW import ──────────────────────────────────────────────────────

TEST_F(ImportTests, Import_SingleRawFile_Succeeds) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("airplane", 1);
  ASSERT_FALSE(images.empty()) << "No test RAW images found in raw/airplane/";

  QSignalSpy importSpy(&backend, &AlbumBackend::ImportStateChanged);
  backend.StartImport(PathsToQStringList(images));

  WaitForImportFinished(backend);

  EXPECT_FALSE(backend.ImportRunning());
  EXPECT_GE(backend.ImportCompleted(), 1);
}

// ── JPEG-only import — must not crash ──────────────────────────────────────

TEST_F(ImportTests, Import_JpegOnly_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Collect .jpg/.jpeg files from the batch folder (they exist alongside ARWs).
  std::vector<std::filesystem::path> jpegPaths;
  const std::filesystem::path batchDir{std::string(TEST_IMG_PATH) + "/raw/batch"};
  if (std::filesystem::exists(batchDir)) {
    for (const auto& entry : std::filesystem::directory_iterator(batchDir)) {
      if (!entry.is_regular_file()) continue;
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg") {
        jpegPaths.push_back(entry.path());
      }
    }
  }
  // Also check jpeg/tile_tests
  const std::filesystem::path jpegDir{std::string(TEST_IMG_PATH) + "/jpeg/tile_tests"};
  if (std::filesystem::exists(jpegDir)) {
    for (const auto& entry : std::filesystem::directory_iterator(jpegDir)) {
      if (entry.is_regular_file()) {
        jpegPaths.push_back(entry.path());
      }
    }
  }

  if (jpegPaths.empty()) {
    GTEST_SKIP() << "No JPEG test images available";
  }

  // This should NOT crash — even if pipeline cannot process JPEGs.
  backend.StartImport(PathsToQStringList(jpegPaths));
  WaitForImportFinished(backend);

  // The test passes as long as we reach here without crashing.
  EXPECT_FALSE(backend.ImportRunning());
}

// ── TIFF-only import — must not crash ──────────────────────────────────────

TEST_F(ImportTests, Import_TiffOnly_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  std::vector<std::filesystem::path> tiffPaths;
  const std::filesystem::path imgRoot{std::string(TEST_IMG_PATH)};
  if (std::filesystem::exists(imgRoot)) {
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(imgRoot)) {
      if (!entry.is_regular_file()) continue;
      auto ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".tiff" || ext == ".tif") {
        tiffPaths.push_back(entry.path());
      }
    }
  }

  if (tiffPaths.empty()) {
    GTEST_SKIP() << "No TIFF test images available";
  }

  backend.StartImport(PathsToQStringList(tiffPaths));
  WaitForImportFinished(backend);

  EXPECT_FALSE(backend.ImportRunning());
}

// ── Mixed RAW + JPEG — only RAWs should survive pipeline without crash ─────

TEST_F(ImportTests, Import_MixedRawAndJpeg_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Collect everything from raw/batch (NEFs, ARWs, JPGs).
  std::vector<std::filesystem::path> allPaths;
  const std::filesystem::path batchDir{std::string(TEST_IMG_PATH) + "/raw/batch"};
  if (std::filesystem::exists(batchDir)) {
    for (const auto& entry : std::filesystem::directory_iterator(batchDir)) {
      if (entry.is_regular_file() && is_supported_file(entry.path())) {
        allPaths.push_back(entry.path());
      }
    }
  }
  // Limit to a reasonable count so the test finishes quickly.
  std::sort(allPaths.begin(), allPaths.end());
  if (allPaths.size() > 6) allPaths.resize(6);

  if (allPaths.empty()) {
    GTEST_SKIP() << "No mixed test images available in raw/batch/";
  }

  backend.StartImport(PathsToQStringList(allPaths));
  WaitForImportFinished(backend);

  EXPECT_FALSE(backend.ImportRunning());
  // At least the RAW files should succeed (if JPEGs fail, that's OK —
  // the critical thing is no crash).
}

// ── Empty file list — no crash, sensible feedback ──────────────────────────

TEST_F(ImportTests, Import_EmptyFileList_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  backend.StartImport({});

  ProcessEvents(200);

  EXPECT_FALSE(backend.ImportRunning());
}

// ── Non-existent path — no crash ───────────────────────────────────────────

TEST_F(ImportTests, Import_NonexistentPath_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  QStringList fakeFiles;
  fakeFiles << "C:/this/path/does/not/exist/photo.nef";
  fakeFiles << "/tmp/phantom_image.arw";

  backend.StartImport(fakeFiles);
  ProcessEvents(200);

  EXPECT_FALSE(backend.ImportRunning());
}

// ── Import without a project loaded — no crash ─────────────────────────────

TEST_F(ImportTests, Import_NoProjectLoaded_NoCrash) {
  AlbumBackend backend;
  // Do NOT create a project — backend is in "not ready" state.

  auto images = CollectRawTestImages("airplane", 1);
  if (images.empty()) {
    GTEST_SKIP() << "No test RAW images available";
  }

  // Should fail gracefully, not crash.
  backend.StartImport(PathsToQStringList(images));
  ProcessEvents(200);

  EXPECT_FALSE(backend.ImportRunning());
  EXPECT_FALSE(backend.ServiceReady());
}

// ── Duplicate files in same import call — deduplication ────────────────────

TEST_F(ImportTests, Import_DuplicateFiles_Deduplication) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("airplane", 1);
  ASSERT_FALSE(images.empty());

  // Duplicate the same file path twice.
  QStringList duped;
  duped << PathToQString(images[0]);
  duped << PathToQString(images[0]);

  backend.StartImport(duped);
  WaitForImportFinished(backend);

  EXPECT_FALSE(backend.ImportRunning());
  // Only one copy should have been imported.
  EXPECT_EQ(backend.ImportCompleted(), 1);
}

// ── Batch DNG import ───────────────────────────────────────────────────────

TEST_F(ImportTests, Import_BatchDng_Succeeds) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("batch_import", 5);
  if (images.empty()) {
    GTEST_SKIP() << "No DNG files in raw/batch_import/";
  }

  backend.StartImport(PathsToQStringList(images));
  WaitForImportFinished(backend, 60000);  // DNG batch can be slow.

  EXPECT_FALSE(backend.ImportRunning());
  EXPECT_EQ(backend.ImportCompleted(), static_cast<int>(images.size()));
}

// ── Cancel import — no crash ───────────────────────────────────────────────

TEST_F(ImportTests, Import_CancelImmediate_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("batch_import", 10);
  if (images.empty()) {
    GTEST_SKIP() << "No DNG files for cancel test";
  }

  backend.StartImport(PathsToQStringList(images));
  // Cancel immediately — the import thread may or may not have started.
  ProcessEvents(50);
  backend.CancelImport();

  WaitForImportFinished(backend, 15000);
  EXPECT_FALSE(backend.ImportRunning());
}

// ── Unsupported extension — silently ignored ───────────────────────────────

TEST_F(ImportTests, Import_UnsupportedExtension_Ignored) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Create a temp file with a made-up extension.
  const auto fakePath = temp_dir_ / "image.xyz";
  {
    std::ofstream ofs(fakePath);
    ofs << "not a real image";
  }

  QStringList list;
  list << PathToQString(fakePath);
  backend.StartImport(list);
  ProcessEvents(200);

  // The file should be silently skipped (not a supported extension).
  EXPECT_FALSE(backend.ImportRunning());
}

// ── Corrupted file with valid extension — no crash ─────────────────────────

TEST_F(ImportTests, Import_CorruptedFileValidExtension_NoCrash) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  // Create a file named .nef but with garbage content.
  const auto corruptPath = temp_dir_ / "corrupt.nef";
  {
    std::ofstream ofs(corruptPath, std::ios::binary);
    ofs << "THIS_IS_NOT_A_REAL_NEF_FILE_JUST_GARBAGE_DATA";
  }

  QStringList list;
  list << PathToQString(corruptPath);
  backend.StartImport(list);
  WaitForImportFinished(backend);

  // Must not crash. Import may report 0 succeeded + 1 failed, or skip it.
  EXPECT_FALSE(backend.ImportRunning());
}

}  // namespace
}  // namespace puerhlab::ui::test
