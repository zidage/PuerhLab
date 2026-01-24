#include "app/thumbnail_service.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <vector>

#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "storage/service/pipeline/pipeline_service.hpp"
#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/profiler/profiler.hpp"

namespace puerhlab {
class ThumbnailServiceTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;
  std::filesystem::path meta_path_;

  void                  SetUp() override {
    TimeProvider::Refresh();
    Exiv2::LogMsg::setLevel(Exiv2::LogMsg::Level::mute);
    db_path_ = std::filesystem::temp_directory_path() / "thumbnail_service_test.db";
    meta_path_ = std::filesystem::temp_directory_path() / "thumbnail_service_test.json";
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
    RegisterAllOperators();
#ifdef EASY_PROFILER_ENABLE
    EASY_PROFILER_ENABLE;
#endif
  }

  void TearDown() override {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
#ifdef EASY_PROFILER_ENABLE
    profiler::dumpBlocksToFile(TEST_PROFILER_OUTPUT_PATH);
    EASY_PROFILER_DISABLE;
#endif
  }
};

TEST_F(ThumbnailServiceTests, DISABLED_GenerateThumbnailAndCallbacks) {
  using namespace std::chrono_literals;

  ProjectService            project(db_path_, meta_path_);
  auto                      fs_service = project.GetSleeveService();
  auto                      img_pool   = project.GetImagePoolService();

  ImportServiceImpl         import_service(fs_service, img_pool);
  std::filesystem::path     img_root_path = {TEST_IMG_PATH "/raw/batch_import"};
  std::vector<image_path_t> paths{};

  for (const auto& entry : std::filesystem::directory_iterator(img_root_path)) {
    if (entry.is_regular_file()) {
      paths.push_back(entry.path());
    }
  }

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();

  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();
  import_job->on_finished_                       = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service.ImportToFolder(paths, L"", {}, import_job);
  ASSERT_NE(import_job, nullptr);
  //   ASSERT_EQ(final_result_future.wait_for(30s), std::future_status::ready);

  auto final_result_value = final_result_future.get();
  EXPECT_EQ(final_result_value.requested_, paths.size());
  EXPECT_EQ(final_result_value.imported_, paths.size());
  EXPECT_EQ(final_result_value.failed_, 0u);

  ASSERT_NE(import_job->import_log_, nullptr);
  auto snapshot = import_job->import_log_->Snapshot();
  ASSERT_EQ(snapshot.created_.size(), paths.size());
  ASSERT_EQ(snapshot.metadata_ok_.size(), paths.size());

  // Get the first file's thumbnail
  const auto file_id          = snapshot.created_.front().element_id_;

  auto       pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());

  auto       scheduler        = std::make_shared<PipelineScheduler>();
  ThumbnailService                thumbnail_service(img_pool, pipeline_service, scheduler);

  std::shared_ptr<ThumbnailGuard> guard_1;
  std::shared_ptr<ThumbnailGuard> guard_2;
  std::atomic<int>                callback_count{0};
  std::promise<void>              done_1;
  std::promise<void>              done_2;
  auto                            f1       = done_1.get_future();
  auto                            f2       = done_2.get_future();

  auto                            callback = [&](std::shared_ptr<ThumbnailGuard> guard) {
    const int idx = callback_count.fetch_add(1);
    if (idx == 0) {
      guard_1 = guard;
      done_1.set_value();
      return;
    }
    if (idx == 1) {
      guard_2 = guard;
      done_2.set_value();
    }
  };

  // Ideally, one of these two requests should hit the pending queue
  thumbnail_service.GetThumbnail(file_id, callback);
  thumbnail_service.GetThumbnail(file_id, callback);

  ASSERT_EQ(f1.wait_for(30s), std::future_status::ready);
  ASSERT_EQ(f2.wait_for(30s), std::future_status::ready);

  ASSERT_NE(guard_1, nullptr);
  ASSERT_NE(guard_2, nullptr);
  EXPECT_EQ(guard_1.get(), guard_2.get());
  ASSERT_NE(guard_1->thumbnail_buffer_, nullptr);

  auto* buffer = guard_1->thumbnail_buffer_.get();
  if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
    EXPECT_NO_THROW(buffer->SyncToCPU());
  }
  if (buffer->cpu_data_valid_) {
    auto& mat = buffer->GetCPUData();
    std::cout << "[ThumbnailTest] thumbnail size: " << mat.cols << "x" << mat.rows
              << " ch=" << mat.channels() << std::endl;
    EXPECT_FALSE(mat.empty());
  } else {
    FAIL() << "Thumbnail buffer has no CPU data";
  }

  EXPECT_EQ(guard_1->pin_count_, 1);

  std::promise<std::shared_ptr<ThumbnailGuard>> cached;
  auto                                          cached_future  = cached.get_future();
  int                                           dispatch_count = 0;
  auto dispatcher = [&dispatch_count](std::function<void()> fn) {
    ++dispatch_count;
    fn();
  };

  thumbnail_service.GetThumbnail(
      file_id, [&cached](std::shared_ptr<ThumbnailGuard> guard) { cached.set_value(guard); }, true,
      dispatcher);

  ASSERT_EQ(cached_future.wait_for(5s), std::future_status::ready);
  auto cached_guard = cached_future.get();
  ASSERT_NE(cached_guard, nullptr);
  EXPECT_EQ(cached_guard.get(), guard_1.get());
  EXPECT_EQ(dispatch_count, 1);
  EXPECT_EQ(cached_guard->pin_count_, 2);

  thumbnail_service.ReleaseThumbnail(file_id);
  EXPECT_EQ(cached_guard->pin_count_, 1);

  std::promise<std::shared_ptr<ThumbnailGuard>> no_pin;
  auto                                          no_pin_future = no_pin.get_future();
  // This time, the thumbnail is already cached but not pinned
  thumbnail_service.GetThumbnail(
      file_id, [&no_pin](std::shared_ptr<ThumbnailGuard> guard) { no_pin.set_value(guard); },
      false);

  ASSERT_EQ(no_pin_future.wait_for(5s), std::future_status::ready);
  auto no_pin_guard = no_pin_future.get();
  ASSERT_NE(no_pin_guard, nullptr);
  EXPECT_EQ(no_pin_guard->pin_count_, 1);

  thumbnail_service.ReleaseThumbnail(file_id);
  EXPECT_EQ(no_pin_guard->pin_count_, 0);
}

TEST_F(ThumbnailServiceTests, Generate16ThumbnailsAndValidateAll) {
  using namespace std::chrono_literals;

  ProjectService            project(db_path_, meta_path_);
  auto                      fs_service = project.GetSleeveService();
  auto                      img_pool   = project.GetImagePoolService();
  ImportServiceImpl         import_service(fs_service, img_pool);

  std::filesystem::path     img_root_path = {TEST_IMG_PATH "/raw/batch_import"};
  std::vector<image_path_t> paths{};
  for (const auto& entry : std::filesystem::directory_iterator(img_root_path)) {
    if (entry.is_regular_file()) {
      paths.push_back(entry.path());
    }
  }

  ASSERT_GE(paths.size(), 16u) << "Need at least 16 images under TEST_IMG_PATH/raw/batch_import";

  std::shared_ptr<ImportJob> import_job = std::make_shared<ImportJob>();
  std::promise<ImportResult> final_result;
  auto                       final_result_future = final_result.get_future();
  import_job->on_finished_                       = [&final_result](const ImportResult& result) {
    final_result.set_value(result);
  };

  import_job = import_service.ImportToFolder(paths, L"", {}, import_job);
  ASSERT_NE(import_job, nullptr);
  ASSERT_EQ(final_result_future.wait_for(60s), std::future_status::ready)
      << "Import did not finish in time";

  auto final_result_value = final_result_future.get();
  EXPECT_EQ(final_result_value.requested_, paths.size());
  EXPECT_EQ(final_result_value.imported_, paths.size());
  EXPECT_EQ(final_result_value.failed_, 0u);

  ASSERT_NE(import_job->import_log_, nullptr);
  auto snapshot = import_job->import_log_->Snapshot();
  ASSERT_GE(snapshot.created_.size(), paths.size());
  ASSERT_GE(snapshot.metadata_ok_.size(), paths.size());

  auto pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto scheduler        = std::make_shared<PipelineScheduler>(8);
  ThumbnailService             thumbnail_service(img_pool, pipeline_service, scheduler);

  std::vector<sl_element_id_t> ids;
  ids.reserve(16);
  // Get first 16 imported images
  for (size_t i = 0; i < 16; ++i) {
    ids.push_back(snapshot.created_[i].element_id_);
  }

  std::vector<std::promise<std::shared_ptr<ThumbnailGuard>>> done_promises(16);
  std::vector<std::future<std::shared_ptr<ThumbnailGuard>>>  done_futures;
  done_futures.reserve(16);
  for (auto& p : done_promises) {
    done_futures.push_back(p.get_future());
  }

  std::vector<std::shared_ptr<ThumbnailGuard>> guards(16);

  for (size_t i = 0; i < 16; ++i) {
    const auto id = ids[i];
    thumbnail_service.GetThumbnail(
        id,
        [i, &guards, &done_promises](std::shared_ptr<ThumbnailGuard> guard) {
          guards[i] = guard;
          done_promises[i].set_value(guard);
        },
        true);
  }

  for (size_t i = 0; i < 16; ++i) {
    ASSERT_EQ(done_futures[i].wait_for(60s), std::future_status::ready)
        << "Thumbnail generation timed out at index " << i;
  }

  for (size_t i = 0; i < 16; ++i) {
    auto guard = done_futures[i].get();
    ASSERT_NE(guard, nullptr) << "Null ThumbnailGuard at index " << i;
    ASSERT_NE(guard->thumbnail_buffer_, nullptr) << "Null ImageBuffer at index " << i;

    auto* buffer = guard->thumbnail_buffer_.get();
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      EXPECT_NO_THROW(buffer->SyncToCPU()) << "SyncToCPU failed at index " << i;
    }

    ASSERT_TRUE(buffer->cpu_data_valid_) << "Thumbnail has no CPU data at index " << i;
    auto& mat = buffer->GetCPUData();
    std::cout << "[Thumbnail16Test] idx=" << i << " size=" << mat.cols << "x" << mat.rows
              << " ch=" << mat.channels() << std::endl;
    EXPECT_FALSE(mat.empty()) << "Empty thumbnail mat at index " << i;

    // We pinned on request; each thumbnail should be pinned at least once.
    EXPECT_GE(guard->pin_count_, 1) << "Unexpected pin_count at index " << i;
  }

  // Different ids should generally produce different guards/buffers.
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = i + 1; j < 16; ++j) {
      EXPECT_NE(guards[i].get(), guards[j].get()) << "Guards unexpectedly shared across ids";
    }
  }

  // Release pins we took.
  for (const auto id : ids) {
    thumbnail_service.ReleaseThumbnail(id);
  }
}

TEST_F(ThumbnailServiceTests, DISABLED_MissingPipelineThrows) {
  ProjectService project(db_path_, meta_path_);
  auto           img_pool        = project.GetImagePoolService();

  auto           storage_service = project.GetStorageService();
  auto           conn_guard      = storage_service->GetDBController().GetConnectionGuard();
  auto pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto scheduler        = std::make_shared<PipelineScheduler>();

  ThumbnailService thumbnail_service(img_pool, pipeline_service, scheduler);

  EXPECT_THROW(thumbnail_service.GetThumbnail(12345, [](std::shared_ptr<ThumbnailGuard>) {}),
               std::runtime_error);
}

TEST_F(ThumbnailServiceTests, DISABLED_MissingImageThrows) {
  ProjectService project(db_path_, meta_path_);
  auto           img_pool        = project.GetImagePoolService();

  auto           storage_service = project.GetStorageService();
  auto           conn_guard      = storage_service->GetDBController().GetConnectionGuard();
  auto pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto scheduler        = std::make_shared<PipelineScheduler>();

  constexpr sl_element_id_t kMissingImageId = 7777;

  ThumbnailService          thumbnail_service(img_pool, pipeline_service, scheduler);

  EXPECT_THROW(
      thumbnail_service.GetThumbnail(kMissingImageId, [](std::shared_ptr<ThumbnailGuard>) {}),
      std::runtime_error);
}
};  // namespace puerhlab
