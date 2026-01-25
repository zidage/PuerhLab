#include "app/thumbnail_service.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "storage/service/pipeline/pipeline_service.hpp"
#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/profiler/profiler.hpp"

namespace puerhlab {

namespace {
using namespace std::chrono_literals;

static uint64_t HashBytesFnv1a64(const uint8_t* data, size_t size) {
  // FNV-1a 64-bit
  constexpr uint64_t kOffset = 14695981039346656037ull;
  constexpr uint64_t kPrime  = 1099511628211ull;
  uint64_t           h       = kOffset;
  for (size_t i = 0; i < size; ++i) {
    h ^= static_cast<uint64_t>(data[i]);
    h *= kPrime;
  }
  return h;
}

static uint64_t HashMatBytes(const cv::Mat& mat) {
  if (mat.empty()) {
    return 0;
  }
  const auto row_bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
  if (mat.isContinuous()) {
    const auto total_bytes = static_cast<size_t>(mat.total()) * mat.elemSize();
    return HashBytesFnv1a64(reinterpret_cast<const uint8_t*>(mat.data), total_bytes);
  }
  uint64_t h = 14695981039346656037ull;
  for (int r = 0; r < mat.rows; ++r) {
    const auto*        row_ptr = mat.ptr<uint8_t>(r);
    // Mix each row into the same FNV stream
    constexpr uint64_t kPrime  = 1099511628211ull;
    for (size_t i = 0; i < row_bytes; ++i) {
      h ^= static_cast<uint64_t>(row_ptr[i]);
      h *= kPrime;
    }
  }
  return h;
}

static std::shared_ptr<ThumbnailGuard> GetThumbnailBlocking(ThumbnailService& service,
                                                            sl_element_id_t id, image_id_t image_id,
                                                            bool pin_if_found = true) {
  std::promise<std::shared_ptr<ThumbnailGuard>> done;
  auto                                          fut = done.get_future();
  service.GetThumbnail(
      id, image_id, [&done](std::shared_ptr<ThumbnailGuard> guard) { done.set_value(guard); },
      pin_if_found);
  EXPECT_EQ(fut.wait_for(60s), std::future_status::ready);
  return fut.get();
}

static uint64_t GetThumbnailHashBlocking(ThumbnailService& service, sl_element_id_t id,
                                         image_id_t image_id) {
  auto guard = GetThumbnailBlocking(service, id, image_id);
  EXPECT_NE(guard, nullptr);
  EXPECT_NE(guard->thumbnail_buffer_, nullptr);

  auto* buffer = guard->thumbnail_buffer_.get();
  if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
    EXPECT_NO_THROW(buffer->SyncToCPU());
  }
  EXPECT_TRUE(buffer->cpu_data_valid_);
  auto& mat = buffer->GetCPUData();
  EXPECT_FALSE(mat.empty());
  return HashMatBytes(mat);
}

static void ReleaseAllThumbnailsAggressively(ThumbnailService& service,
                                            const std::vector<std::pair<sl_element_id_t, image_id_t>>& ids,
                                            int release_rounds = 32) {
  for (int r = 0; r < release_rounds; ++r) {
    for (const auto& [id, image_id] : ids) {
      (void)image_id;
      service.ReleaseThumbnail(id);
    }
  }
}

static void FuzzScrollRequestsNoThrow(
    ThumbnailService& service,
    const std::vector<std::pair<sl_element_id_t, image_id_t>>& ids,
    size_t iterations,
    uint32_t seed) {
  ASSERT_FALSE(ids.empty());

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> step_dist(-3, 3);
  std::uniform_int_distribution<int> pct_dist(0, 99);

  auto clamp_index = [&](int idx) -> size_t {
    if (idx < 0) {
      return 0;
    }
    const int max_idx = static_cast<int>(ids.size()) - 1;
    if (idx > max_idx) {
      return static_cast<size_t>(max_idx);
    }
    return static_cast<size_t>(idx);
  };

  auto reflect_index = [&](int idx) -> size_t {
    if (ids.size() == 1) {
      return 0;
    }
    const int max_idx = static_cast<int>(ids.size()) - 1;
    if (idx < 0) {
      idx = -idx;
    }
    if (idx > max_idx) {
      idx = (2 * max_idx) - idx;
      if (idx < 0) {
        idx = 0;
      }
    }
    return static_cast<size_t>(idx);
  };

  size_t pos = static_cast<size_t>(rng() % ids.size());

  std::vector<std::future<std::shared_ptr<ThumbnailGuard>>> futures;
  futures.reserve(iterations * 3);

  for (size_t i = 0; i < iterations; ++i) {
    int step = step_dist(rng);
    if (step == 0) {
      step = (pct_dist(rng) < 50) ? 1 : -1;
    }

    // Simulate "scrolling" and bouncing at the ends.
    pos = reflect_index(static_cast<int>(pos) + step);

    // Request current item plus neighbors (simple prefetch).
    const int offsets[3] = {0, 1, -1};
    for (const int off : offsets) {
      const size_t idx = clamp_index(static_cast<int>(pos) + off);
      const auto   id = ids[idx].first;
      const auto   image_id = ids[idx].second;
      const bool   pin = (pct_dist(rng) < 60);

      auto promise = std::make_shared<std::promise<std::shared_ptr<ThumbnailGuard>>>();
      futures.push_back(promise->get_future());

      EXPECT_NO_THROW(service.GetThumbnail(
          id, image_id,
          [promise](std::shared_ptr<ThumbnailGuard> guard) {
            // Guard against multiple callbacks/promise already satisfied.
            try {
              promise->set_value(guard);
            } catch (...) {
            }
          },
          pin));
    }

    // Randomly release around the current position to emulate the user moving away.
    if (pct_dist(rng) < 35) {
      const auto id = ids[pos].first;
      EXPECT_NO_THROW(service.ReleaseThumbnail(id));
    }
    if (pct_dist(rng) < 15) {
      const size_t idx = clamp_index(static_cast<int>(pos) + ((pct_dist(rng) < 50) ? 2 : -2));
      EXPECT_NO_THROW(service.ReleaseThumbnail(ids[idx].first));
    }
  }

  // Validate completion and that we didn't crash/throw while producing thumbnails.
  for (auto& fut : futures) {
    ASSERT_EQ(fut.wait_for(60s), std::future_status::ready) << "Thumbnail request timed out";
    auto guard = fut.get();
    ASSERT_NE(guard, nullptr);
    ASSERT_NE(guard->thumbnail_buffer_, nullptr);

    auto* buffer = guard->thumbnail_buffer_.get();
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      EXPECT_NO_THROW(buffer->SyncToCPU());
    }
    ASSERT_TRUE(buffer->cpu_data_valid_);
    auto& mat = buffer->GetCPUData();
    EXPECT_FALSE(mat.empty());
  }
}
}  // namespace

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

TEST_F(ThumbnailServiceTests, GenerateThumbnailAndCallbacks) {
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
  const auto image_id         = snapshot.created_.front().image_id_;

  auto       pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());

  ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

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
  thumbnail_service.GetThumbnail(file_id, image_id, callback);
  thumbnail_service.GetThumbnail(file_id, image_id, callback);

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
      file_id, image_id,
      [&cached](std::shared_ptr<ThumbnailGuard> guard) { cached.set_value(guard); }, true,
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
      file_id, image_id,
      [&no_pin](std::shared_ptr<ThumbnailGuard> guard) { no_pin.set_value(guard); }, false);

  ASSERT_EQ(no_pin_future.wait_for(5s), std::future_status::ready);
  auto no_pin_guard = no_pin_future.get();
  ASSERT_NE(no_pin_guard, nullptr);
  EXPECT_EQ(no_pin_guard->pin_count_, 1);

  thumbnail_service.ReleaseThumbnail(file_id);
  EXPECT_EQ(no_pin_guard->pin_count_, 0);
}

TEST_F(ThumbnailServiceTests, PipelineRestoredFromDBGeneratesCorrectThumbnail) {
  // Verify that thumbnail generation uses the pipeline restored from DB (not only the fresh
  // default pipeline created after app init).

  sl_element_id_t file_id  = 0;
  image_id_t      image_id = 0;

  // Phase 1: import and get a file id.
  {
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
    ASSERT_FALSE(paths.empty()) << "Need at least 1 image under TEST_IMG_PATH/raw/batch_import";

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

    const auto final_result_value = final_result_future.get();
    ASSERT_EQ(final_result_value.failed_, 0u);
    ASSERT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    ASSERT_FALSE(snapshot.created_.empty());
    file_id  = snapshot.created_.front().element_id_;

    image_id = snapshot.created_.front().image_id_;
    ASSERT_NE(file_id, 0);

    import_service.SyncImports(snapshot, L"");

    project.GetSleeveService()->Sync();
    project.GetImagePoolService()->SyncWithStorage();

    project.SaveProject(meta_path_);
  }

  uint64_t default_hash  = 0;
  uint64_t modified_hash = 0;

  // Phase 2: generate default thumbnail, then modify pipeline, persist, and ensure thumbnail
  // changes.
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    std::string pipline_before;
    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);
      default_hash = GetThumbnailHashBlocking(thumbnail_service, file_id, image_id);
      thumbnail_service.ReleaseThumbnail(file_id);

      auto pipeline = pipeline_service->LoadPipeline(file_id);
      pipline_before = pipeline->pipeline_->ExportPipelineParams().dump();
      pipeline_service->SavePipeline(pipeline);
    }

    auto pipline_after = std::string{};
    // Modify pipeline parameters and persist to DB.
    {
      auto guard = pipeline_service->LoadPipeline(file_id);
      ASSERT_NE(guard, nullptr);
      auto&          stage = guard->pipeline_->GetStage(PipelineStageName::Basic_Adjustment);
      nlohmann::json params;
      // Use a strong exposure change so the thumbnail content should differ.
      params["exposure"] = 3.0f;
      stage.SetOperator(OperatorType::EXPOSURE, params, guard->pipeline_->GetGlobalParams());
      guard->dirty_ = true;
      pipeline_service->SavePipeline(guard);
    }

    // New ThumbnailService instance to avoid serving the old cached thumbnail.
    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);
      auto             pipeline = pipeline_service->LoadPipeline(file_id);
      pipline_after = pipeline->pipeline_->ExportPipelineParams().dump();
      ASSERT_NE(pipline_before, pipline_after) << "Pipeline parameters did not change after modification";
      modified_hash = GetThumbnailHashBlocking(thumbnail_service, file_id, image_id);
      thumbnail_service.ReleaseThumbnail(file_id);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }

  ASSERT_NE(default_hash, 0ull);
  ASSERT_NE(modified_hash, 0ull);
  EXPECT_NE(modified_hash, default_hash) << "Pipeline change did not affect generated thumbnail";

  // Phase 3: reopen project (pipeline restored from DB) and ensure thumbnail matches the modified
  // one.
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);
    const auto       restored_hash = GetThumbnailHashBlocking(thumbnail_service, file_id, image_id);
    thumbnail_service.ReleaseThumbnail(file_id);

    EXPECT_EQ(restored_hash, modified_hash)
        << "Restored pipeline did not produce the expected thumbnail";
  }
}

TEST_F(ThumbnailServiceTests, FuzzScrollBrowsingNoThrowReloadService) {
  // Simulate a user scrolling back and forth in the UI thumbnail grid.
  // Requirement: no throws; two phases; and on each "service shutdown" persist like
  // PipelineRestoredFromDBGeneratesCorrectThumbnail.

  // Phase 0: import images once and persist the project.
  std::vector<std::pair<sl_element_id_t, image_id_t>> ids;
  {
    ProjectService        project(db_path_, meta_path_);
    auto                  fs_service = project.GetSleeveService();
    auto                  img_pool   = project.GetImagePoolService();
    ImportServiceImpl     import_service(fs_service, img_pool);
    std::filesystem::path img_root_path = {TEST_IMG_PATH "/raw/batch_import"};

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

    const auto final_result_value = final_result_future.get();
    ASSERT_EQ(final_result_value.failed_, 0u);
    ASSERT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    ASSERT_GE(snapshot.created_.size(), 16u);

    ids.reserve(32);
    const size_t count = std::min<size_t>(32, snapshot.created_.size());
    for (size_t i = 0; i < count; ++i) {
      ids.push_back({snapshot.created_[i].element_id_, snapshot.created_[i].image_id_});
    }

    import_service.SyncImports(snapshot, L"");
    project.GetSleeveService()->Sync();
    project.GetImagePoolService()->SyncWithStorage();
    project.SaveProject(meta_path_);
  }

  ASSERT_FALSE(ids.empty());

  // Phase 1: browse/fuzz.
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);
      FuzzScrollRequestsNoThrow(thumbnail_service, ids, 250, 0xC0FFEEu);
      ReleaseAllThumbnailsAggressively(thumbnail_service, ids);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }

  // Phase 2: simulate reloading the service and browsing again.
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);
      FuzzScrollRequestsNoThrow(thumbnail_service, ids, 250, 0xBADC0DEu);
      ReleaseAllThumbnailsAggressively(thumbnail_service, ids);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }
}

TEST_F(ThumbnailServiceTests, Generate16ThumbnailsAndValidateAll) {
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
  ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

  std::vector<std::pair<sl_element_id_t, image_id_t>> ids;
  ids.reserve(16);
  // Get first 16 imported images
  for (size_t i = 0; i < 16; ++i) {
    ids.push_back({snapshot.created_[i].element_id_, snapshot.created_[i].image_id_});
  }

  std::vector<std::promise<std::shared_ptr<ThumbnailGuard>>> done_promises(16);
  std::vector<std::future<std::shared_ptr<ThumbnailGuard>>>  done_futures;
  done_futures.reserve(16);
  for (auto& p : done_promises) {
    done_futures.push_back(p.get_future());
  }

  std::vector<std::shared_ptr<ThumbnailGuard>> guards(16);

  for (size_t i = 0; i < 16; ++i) {
    const auto [id, image_id] = ids[i];
    thumbnail_service.GetThumbnail(
        id, image_id, [i, &guards, &done_promises](std::shared_ptr<ThumbnailGuard> guard) {
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
  for (const auto& [id, image_id] : ids) {
    thumbnail_service.ReleaseThumbnail(id);
  }
}

TEST_F(ThumbnailServiceTests, MissingPipelineThrows) {
  ProjectService project(db_path_, meta_path_);
  auto           img_pool        = project.GetImagePoolService();

  auto           storage_service = project.GetStorageService();
  auto           conn_guard      = storage_service->GetDBController().GetConnectionGuard();
  auto pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto scheduler        = std::make_shared<PipelineScheduler>();

  ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

  EXPECT_THROW(thumbnail_service.GetThumbnail(12345, 12345, [](std::shared_ptr<ThumbnailGuard>) {}),
               std::runtime_error);
}

TEST_F(ThumbnailServiceTests, MissingImageThrows) {
  ProjectService project(db_path_, meta_path_);
  auto           img_pool        = project.GetImagePoolService();

  auto           storage_service = project.GetStorageService();
  auto           conn_guard      = storage_service->GetDBController().GetConnectionGuard();
  auto pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto scheduler        = std::make_shared<PipelineScheduler>();

  constexpr sl_element_id_t kMissingImageId = 7777;

  ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

  EXPECT_THROW(thumbnail_service.GetThumbnail(kMissingImageId, kMissingImageId,
                                              [](std::shared_ptr<ThumbnailGuard>) {}),
               std::runtime_error);
}
};  // namespace puerhlab
