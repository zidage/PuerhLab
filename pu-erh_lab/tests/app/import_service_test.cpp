#include "app/import_service.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "app/project_service.hpp"
#include "import_test_fixation.hpp"
#include "type/type.hpp"

namespace puerhlab {
TEST_F(ImportServiceTests, InitTest) {
  {
    auto              img_pool = std::make_shared<ImagePoolManager>(128, 4);

    ProjectService    project(db_path_, meta_path_, 0);
    auto              fs_service = project.GetSleeveService();

    EXPECT_NO_THROW(std::unique_ptr<ImportService> import_service =
                        std::make_unique<ImportServiceImpl>(fs_service, img_pool));
  }
}

TEST_F(ImportServiceTests, ImportEmptyTest) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();

  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t>  empty_paths;

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

  ImportResult               final_result;
  bool                       job_finished = false;

  import_job->on_finished_ = [&final_result, &job_finished](const ImportResult& result) {
    final_result = result;
    job_finished = true;
  };

  import_job = import_service->ImportToFolder(empty_paths, L"", {}, import_job);

  ASSERT_NE(import_job, nullptr);

  // Wait for the job to finish
  while (!job_finished) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  EXPECT_EQ(final_result.requested_, 0);
  EXPECT_EQ(final_result.imported_, 0);
  EXPECT_EQ(final_result.failed_, 0);
}

TEST_F(ImportServiceTests, ImportSingleFileTest) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();

  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;
  paths.push_back(TEST_IMG_PATH "/raw/airplane/_DSC1704.NEF");

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();

  import_job->on_progress_                       = [](const ImportProgress& progress) {
    // Can log progress here if needed
    std::cout << "Progress: " << progress.metadata_done_ << "/" << progress.total_ << "\n";
  };

  import_job->on_finished_ = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

  ASSERT_NE(import_job, nullptr);

  // Wait for the job to finish
  final_result_future.wait();

  ImportResult final_result_value = final_result_future.get();

  EXPECT_EQ(final_result_value.requested_, 1);
  EXPECT_EQ(final_result_value.imported_, 1);
  EXPECT_EQ(final_result_value.failed_, 0);

  ASSERT_NE(import_job->import_log_, nullptr);
  auto snapshot = import_job->import_log_->Snapshot();
  EXPECT_EQ(snapshot.created_.size(), 1u);
  EXPECT_EQ(snapshot.metadata_ok_.size(), 1u);
  EXPECT_EQ(snapshot.metadata_failed_.size(), 0u);

  import_service->SyncImports(snapshot, L"");
}

TEST_F(ImportServiceTests, ImportInvalidFileTest) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();

  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;
  paths.push_back(TEST_IMG_PATH "/raw/airplane/invalid_file.txt");

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();

  import_job->on_progress_                       = [](const ImportProgress& progress) {
    // Can log progress here if needed
    std::cout << "Progress: " << progress.metadata_done_ << "/" << progress.total_ << "\n";
  };

  import_job->on_finished_ = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

  ASSERT_NE(import_job, nullptr);

  // Wait for the job to finish
  final_result_future.wait();

  ImportResult final_result_value = final_result_future.get();

  EXPECT_EQ(final_result_value.requested_, 1);
  EXPECT_EQ(final_result_value.imported_, 0);
  EXPECT_EQ(final_result_value.failed_, 1);

  ASSERT_NE(import_job->import_log_, nullptr);
  auto snapshot = import_job->import_log_->Snapshot();
  EXPECT_EQ(snapshot.created_.size(), 0u);
  EXPECT_EQ(snapshot.metadata_ok_.size(), 0u);
  EXPECT_EQ(snapshot.metadata_failed_.size(), 0u);

  EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
}

static auto on_progress_logger = [](const ImportProgress& progress) {
  const uint32_t     total     = progress.total_ ? progress.total_ : 1;  // avoid div by zero
  const uint32_t     done      = progress.metadata_done_;
  const uint32_t     pct       = static_cast<uint32_t>((done * 100) / total);
  constexpr uint32_t bar_width = 24;
  const uint32_t     filled    = (pct * bar_width) / 100;
  std::string        bar(bar_width, ' ');
  for (uint32_t i = 0; i < filled; ++i) {
    bar[i] = '#';
  }

  // Clear line, show green progress bar with percentage
  std::cout << "\r\033[2K[" << bar << "] " << pct << "%"
            << " | " << done << "/" << total << std::flush;
};

TEST_F(ImportServiceTests, ImportWithNonExistentFiles) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();
  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;
  paths.push_back(TEST_IMG_PATH "/raw/airplane/non_existent_file.NEF");

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();

  import_job->on_progress_                       = [](const ImportProgress& progress) {
    // Can log progress here if needed
    std::cout << "Progress: " << progress.metadata_done_ << "/" << progress.total_ << "\n";
  };

  import_job->on_finished_ = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

  ASSERT_NE(import_job, nullptr);

  // Wait for the job to finish
  final_result_future.wait();

  ImportResult final_result_value = final_result_future.get();

  EXPECT_EQ(final_result_value.requested_, 1);
  EXPECT_EQ(final_result_value.imported_, 0);
  EXPECT_EQ(final_result_value.failed_, 1);

  auto snapshot = import_job->import_log_->Snapshot();

  EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
}

TEST_F(ImportServiceTests, BatchReadTest) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);
  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();
  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;

  image_path_t              img_dir = TEST_IMG_PATH "/raw/batch_import";
  for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
    paths.push_back(entry.path());
  }

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();
  import_job->on_progress_                       = on_progress_logger;

  import_job->on_finished_                       = [&final_result](const ImportResult& result) {
    std::cout << std::endl;
    final_result.set_value(result);
  };
  import_job = import_service->ImportToFolder(paths, L"", {}, import_job);
  ASSERT_NE(import_job, nullptr);
  // Wait for the job to finish
  final_result_future.wait();
  ImportResult final_result_value = final_result_future.get();
  EXPECT_EQ(final_result_value.requested_, static_cast<uint32_t>(paths.size()));
  EXPECT_EQ(final_result_value.imported_, static_cast<uint32_t>(paths.size()));
  EXPECT_EQ(final_result_value.failed_, 0);

  auto snapshot = import_job->import_log_->Snapshot();
  EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
}

TEST_F(ImportServiceTests, BatchCancelTest) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();
  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;

  image_path_t              img_dir = TEST_IMG_PATH "/raw/batch_import";
  for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
    paths.push_back(entry.path());
  }

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();
  import_job->on_progress_                       = on_progress_logger;
  import_job->on_finished_                       = [&final_result](const ImportResult& result) {
    std::cout << std::endl;
    final_result.set_value(result);
  };
  import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

  // Sleep for a short while and then cancel
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  import_job->canceled_.store(true);

  ASSERT_NE(import_job, nullptr);
  // Wait for the job to finish
  final_result_future.wait();
  ImportResult final_result_value = final_result_future.get();
  EXPECT_EQ(final_result_value.requested_, static_cast<uint32_t>(paths.size()));
  EXPECT_LT(final_result_value.imported_, static_cast<uint32_t>(paths.size()));
  EXPECT_EQ(final_result_value.failed_, 0);
  std::cout << "This test may fail occasionally in Release mode due to fast imports." << std::endl;

  auto snapshot = import_job->import_log_->Snapshot();
  EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
}

TEST_F(ImportServiceTests, ImportWithInvalidFilesTest) {
  // Create several text files to the test directory
  image_path_t  img_dir           = TEST_IMG_PATH "/raw/batch_import";
  constexpr int num_invalid_files = 5;

  for (int i = 0; i < num_invalid_files; ++i) {
    std::ofstream ofs(img_dir.string() + "/invalid_file_" + std::to_string(i) + ".txt");
    ofs << "This is an invalid image file." << std::endl;
    ofs.close();
  }

  try {
    auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

    std::vector<image_path_t>      paths;
    ProjectService                 project(db_path_, meta_path_, 0);
    auto                           fs_service = project.GetSleeveService();
    std::unique_ptr<ImportService> import_service =
        std::make_unique<ImportServiceImpl>(fs_service, img_pool);
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
      paths.push_back(entry.path());
    }
    std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
    std::promise<ImportResult> final_result;
    auto                       final_result_future = final_result.get_future();
    import_job->on_progress_                       = on_progress_logger;

    import_job->on_finished_                       = [&final_result](const ImportResult& result) {
      std::cout << std::endl;
      final_result.set_value(result);
    };
    import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

    ASSERT_NE(import_job, nullptr);
    // Wait for the job to finish
    final_result_future.wait();
    ImportResult final_result_value = final_result_future.get();
    EXPECT_EQ(final_result_value.requested_, static_cast<uint32_t>(paths.size()));
    EXPECT_EQ(final_result_value.imported_,
              static_cast<uint32_t>(paths.size()) - num_invalid_files);
    EXPECT_EQ(final_result_value.failed_, num_invalid_files);

    ASSERT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
  } catch (const std::exception& e) {
    FAIL() << "Failed to create SleeveManager: " << e.what();
  }

  // Clean up the created text files
  for (int i = 0; i < num_invalid_files; ++i) {
    std::filesystem::remove(img_dir.string() + "/invalid_file_" + std::to_string(i) + ".txt");
  }
}

TEST_F(ImportServiceTests, ImportPartialSuccessWithMockRawFiles) {
  image_path_t  img_dir                = TEST_IMG_PATH "/raw/batch_import";
  constexpr int num_mock_invalid_files = 3;

  for (int i = 0; i < num_mock_invalid_files; ++i) {
    std::ofstream ofs(img_dir.string() + "/mock_invalid_" + std::to_string(i) + ".NEF");
    ofs << "Mock RAW file with invalid content." << std::endl;
    ofs.close();
  }

  try {
    auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

    std::vector<image_path_t>      paths;
    ProjectService                 project(db_path_, meta_path_, 0);
    auto                           fs_service = project.GetSleeveService();
    std::unique_ptr<ImportService> import_service =
        std::make_unique<ImportServiceImpl>(fs_service, img_pool);
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
      paths.push_back(entry.path());
    }
    std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
    std::promise<ImportResult> final_result;
    auto                       final_result_future = final_result.get_future();
    import_job->on_progress_                       = on_progress_logger;

    import_job->on_finished_                       = [&final_result](const ImportResult& result) {
      std::cout << std::endl;
      final_result.set_value(result);
    };
    import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

    ASSERT_NE(import_job, nullptr);
    final_result_future.wait();
    ImportResult final_result_value = final_result_future.get();

    EXPECT_EQ(final_result_value.requested_, static_cast<uint32_t>(paths.size()));
    EXPECT_EQ(final_result_value.imported_,
              static_cast<uint32_t>(paths.size()) - num_mock_invalid_files);
    EXPECT_EQ(final_result_value.failed_, num_mock_invalid_files);

    ASSERT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    EXPECT_EQ(snapshot.created_.size(), static_cast<uint32_t>(paths.size()));
    EXPECT_EQ(snapshot.metadata_ok_.size(),
              static_cast<uint32_t>(paths.size()) - num_mock_invalid_files);
    EXPECT_EQ(snapshot.metadata_failed_.size(), num_mock_invalid_files);
    EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
  } catch (const std::exception& e) {
    FAIL() << "Failed to create SleeveManager: " << e.what();
  }


  for (int i = 0; i < num_mock_invalid_files; ++i) {
    std::filesystem::remove(img_dir.string() + "/mock_invalid_" + std::to_string(i) + ".NEF");
  }
}

TEST_F(ImportServiceTests, ImportWithDirectories) {
  // Create several directories
  image_path_t  img_dir  = TEST_IMG_PATH "/raw/batch_import";
  constexpr int num_dirs = 3;
  for (int i = 0; i < num_dirs; ++i) {
    std::filesystem::create_directory(img_dir.string() + "/test_dir_" + std::to_string(i));
  }

  try {
    auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

    std::vector<image_path_t>      paths;
    ProjectService                 project(db_path_, meta_path_, 0);
    auto                           fs_service = project.GetSleeveService();
    std::unique_ptr<ImportService> import_service =
        std::make_unique<ImportServiceImpl>(fs_service, img_pool);
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
      paths.push_back(entry.path());
    }

    std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
    std::promise<ImportResult> final_result;
    auto                       final_result_future = final_result.get_future();
    import_job->on_progress_                       = on_progress_logger;

    import_job->on_finished_                       = [&final_result](const ImportResult& result) {
      std::cout << std::endl;
      final_result.set_value(result);
    };
    import_job = import_service->ImportToFolder(paths, L"", {}, import_job);

    ASSERT_NE(import_job, nullptr);
    // Wait for the job to finish
    final_result_future.wait();
    ImportResult final_result_value = final_result_future.get();
    EXPECT_EQ(final_result_value.requested_, static_cast<uint32_t>(paths.size()));
    EXPECT_EQ(final_result_value.imported_, static_cast<uint32_t>(paths.size()) - num_dirs);
    EXPECT_EQ(final_result_value.failed_, num_dirs);

    auto snapshot = import_job->import_log_->Snapshot();
    EXPECT_NO_THROW(import_service->SyncImports(snapshot, L""));
  } catch (const std::exception& e) {
    FAIL() << "Failed to create SleeveManager: " << e.what();
  }

  // Clean up the created directories
  for (int i = 0; i < num_dirs; ++i) {
    std::filesystem::remove(img_dir.string() + "/test_dir_" + std::to_string(i));
  }
}

TEST_F(ImportServiceTests, ImportToNonExistentDestination) {
  auto                           img_pool = std::make_shared<ImagePoolManager>(128, 4);

  ProjectService                 project(db_path_, meta_path_, 0);
  auto                           fs_service = project.GetSleeveService();
  std::unique_ptr<ImportService> import_service =
      std::make_unique<ImportServiceImpl>(fs_service, img_pool);

  std::vector<image_path_t> paths;
  paths.push_back(TEST_IMG_PATH "/raw/airplane/_DSC1704.NEF");

  image_path_t               non_existent_dest = TEST_IMG_PATH "/raw/non_existent_folder";

  std::shared_ptr<ImportJob> import_job        = std::make_shared<ImportJob>();

  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();

  import_job->on_finished_                       = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service->ImportToFolder(paths, non_existent_dest, {}, import_job);

  ASSERT_NE(import_job, nullptr);

  // Wait for the job to finish
  final_result_future.wait();

  ImportResult final_result_value = final_result_future.get();

  EXPECT_EQ(final_result_value.requested_, 1);
  EXPECT_EQ(final_result_value.imported_, 0);
  EXPECT_EQ(final_result_value.failed_, 1);

  auto snapshot = import_job->import_log_->Snapshot();
  EXPECT_NO_THROW(import_service->SyncImports(snapshot, non_existent_dest));
}

};  // namespace puerhlab
