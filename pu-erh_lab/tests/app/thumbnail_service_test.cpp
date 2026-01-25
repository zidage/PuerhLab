#include "app/thumbnail_service.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_set>
#include <vector>

#include "app/import_service.hpp"
#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "app/sleeve_service.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "renderer/pipeline_scheduler.hpp"
#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

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

static void ReleaseAllThumbnailsAggressively(
    ThumbnailService& service, const std::vector<std::pair<sl_element_id_t, image_id_t>>& ids,
    int release_rounds = 32) {
  for (int r = 0; r < release_rounds; ++r) {
    for (const auto& [id, image_id] : ids) {
      (void)image_id;
      service.ReleaseThumbnail(id);
    }
  }
}

static void DrainAndValidateFuture(ThumbnailService& service, sl_element_id_t element_id,
                                   std::future<std::shared_ptr<ThumbnailGuard>> fut,
                                   size_t idx_for_debug, size_t heavy_validate_every,
                                   size_t completed_count, bool auto_release_pin) {
  ASSERT_EQ(fut.wait_for(60s), std::future_status::ready)
      << "Thumbnail request timed out (completed=" << completed_count << ", idx=" << idx_for_debug
      << ")";

  auto guard = fut.get();
  ASSERT_NE(guard, nullptr);
  ASSERT_NE(guard->thumbnail_buffer_, nullptr);

  if (heavy_validate_every != 0 && (completed_count % heavy_validate_every) == 0) {
    auto* buffer = guard->thumbnail_buffer_.get();
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      EXPECT_NO_THROW(buffer->SyncToCPU());
    }
    ASSERT_TRUE(buffer->cpu_data_valid_);
    auto& mat = buffer->GetCPUData();
    EXPECT_FALSE(mat.empty());
  }

  // Simulate the UI cell being recycled: once we're done with this thumbnail, release it.
  // Note: GetThumbnail() always returns a guard for generation, even if pin_if_found=false.
  // Releasing is safe here because ReleaseThumbnail() guards against going below zero.
  if (auto_release_pin) {
    EXPECT_NO_THROW(service.ReleaseThumbnail(element_id));
  }
}

static void FuzzScrollRequestsNoThrow(
    ThumbnailService& service, const std::vector<std::pair<sl_element_id_t, image_id_t>>& ids,
    size_t iterations, uint32_t seed, size_t max_in_flight = 256,
    size_t                                      heavy_validate_every = 2000,
    std::vector<std::weak_ptr<ThumbnailGuard>>* first_seen_guards    = nullptr,
    bool enable_progress = true, size_t progress_every = 500, bool auto_release_pin = true) {
  ASSERT_FALSE(ids.empty());
  if (first_seen_guards) {
    ASSERT_EQ(first_seen_guards->size(), ids.size());
  }

  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> step_dist(-3, 3);
  std::uniform_int_distribution<int> pct_dist(0, 99);

  auto                               clamp_index = [&](int idx) -> size_t {
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

  struct InFlight {
    size_t                                       idx_        = 0;
    sl_element_id_t                              element_id_ = 0;
    std::future<std::shared_ptr<ThumbnailGuard>> future_;
  };

  std::deque<InFlight> in_flight;
  size_t               drained        = 0;

  auto                 print_progress = [&](size_t iter) {
    if (!enable_progress) {
      return;
    }
    if (progress_every == 0) {
      return;
    }
    if (iter == 0 || (iter % progress_every) != 0) {
      return;
    }
    const double pct = (iterations == 0)
                                           ? 100.0
                                           : (100.0 * static_cast<double>(iter) / static_cast<double>(iterations));
    std::cout << "\r\033[2K"
              << "[ThumbnailFuzz] iter=" << iter << "/" << iterations << " ("
              << static_cast<int>(pct) << "%)"
              << " drained=" << drained << " inflight=" << in_flight.size() << std::flush;
  };

  for (size_t i = 0; i < iterations; ++i) {
    print_progress(i);

    int step = step_dist(rng);
    if (step == 0) {
      step = (pct_dist(rng) < 50) ? 1 : -1;
    }

    // Simulate "scrolling" and bouncing at the ends.
    pos                  = reflect_index(static_cast<int>(pos) + step);

    // Request current item plus neighbors (simple prefetch).
    const int offsets[3] = {0, 1, -1};
    for (const int off : offsets) {
      const size_t idx      = clamp_index(static_cast<int>(pos) + off);
      const auto   id       = ids[idx].first;
      const auto   image_id = ids[idx].second;
      const bool   pin      = (pct_dist(rng) < 60);

      auto         promise  = std::make_shared<std::promise<std::shared_ptr<ThumbnailGuard>>>();
      auto         fut      = promise->get_future();

      EXPECT_NO_THROW(service.GetThumbnail(
          id, image_id,
          [promise, first_seen_guards, idx](std::shared_ptr<ThumbnailGuard> guard) {
            // Guard against multiple callbacks/promise already satisfied.
            try {
              if (first_seen_guards && (*first_seen_guards)[idx].expired()) {
                (*first_seen_guards)[idx] = guard;
              }
              promise->set_value(guard);
            } catch (...) {
            }
          },
          pin));

      in_flight.push_back({idx, id, std::move(fut)});
      if (max_in_flight != 0 && in_flight.size() > max_in_flight) {
        auto front = std::move(in_flight.front());
        in_flight.pop_front();
        ++drained;
        DrainAndValidateFuture(service, front.element_id_, std::move(front.future_), front.idx_,
                               heavy_validate_every, drained, auto_release_pin);
      }
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

  // Drain remaining callbacks.
  while (!in_flight.empty()) {
    auto front = std::move(in_flight.front());
    in_flight.pop_front();
    ++drained;
    DrainAndValidateFuture(service, front.element_id_, std::move(front.future_), front.idx_,
                           heavy_validate_every, drained, auto_release_pin);
  }

  if (enable_progress) {
    std::cout << "\r\033[2K" << "[ThumbnailFuzz] iter=" << iterations << "/" << iterations
              << " (100%)"
              << " drained=" << drained << " inflight=0" << std::endl;
  }
}

static void WaitAndValidateFuture(std::future<std::shared_ptr<ThumbnailGuard>> fut,
                                  size_t idx_for_debug, size_t heavy_validate_every,
                                  size_t                           completed_count,
                                  std::shared_ptr<ThumbnailGuard>* out_guard) {
  ASSERT_NE(out_guard, nullptr);
  ASSERT_EQ(fut.wait_for(60s), std::future_status::ready)
      << "Thumbnail request timed out (completed=" << completed_count << ", idx=" << idx_for_debug
      << ")";

  auto guard = fut.get();
  ASSERT_NE(guard, nullptr);
  ASSERT_NE(guard->thumbnail_buffer_, nullptr);

  if (heavy_validate_every != 0 && (completed_count % heavy_validate_every) == 0) {
    auto* buffer = guard->thumbnail_buffer_.get();
    if (!buffer->cpu_data_valid_ && buffer->gpu_data_valid_) {
      EXPECT_NO_THROW(buffer->SyncToCPU());
    }
    ASSERT_TRUE(buffer->cpu_data_valid_);
    auto& mat = buffer->GetCPUData();
    EXPECT_FALSE(mat.empty());
  }

  *out_guard = std::move(guard);
}

// More UI-faithful model: a fixed grid of `view_size` recyclable cells.
// - Each cell is (re)bound to an element idx when scrolling.
// - Binding requests the thumbnail (pin=true) and holds the returned guard.
// - When a cell is rebound (item scrolled away), we ReleaseThumbnail() immediately.
// - Late callbacks for items that are no longer bound are released on completion.
static void FuzzAlbumScrollGridCellsNoThrow(
    ThumbnailService& service, const std::vector<std::pair<sl_element_id_t, image_id_t>>& ids,
    size_t iterations, uint32_t seed, size_t view_size = 50,
    size_t prefetch_each_side = 7,  // 50 + 7*2 = 64 (matches service cache)
    size_t max_in_flight = 12, size_t heavy_validate_every = 5000,
    std::vector<std::weak_ptr<ThumbnailGuard>>* first_seen_guards = nullptr,
    bool enable_progress = true, size_t progress_every = 1000) {
  ASSERT_FALSE(ids.empty());
  if (first_seen_guards) {
    ASSERT_EQ(first_seen_guards->size(), ids.size());
  }

  const size_t                       window    = std::min(view_size, ids.size());
  const size_t                       max_start = (ids.size() > window) ? (ids.size() - window) : 0;

  std::mt19937                       rng(seed);
  std::uniform_int_distribution<int> step_dist(-8, 8);
  std::uniform_int_distribution<int> pct_dist(0, 99);

  auto                               reflect_center = [&](int idx) -> size_t {
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

  struct InFlight {
    size_t                                       idx_              = 0;
    sl_element_id_t                              element_id_       = 0;
    bool                                         requested_pinned_ = false;
    std::future<std::shared_ptr<ThumbnailGuard>> future_;
  };

  std::deque<InFlight>                         in_flight;
  std::unordered_set<size_t>                   in_flight_idx;

  // A fixed set of UI cells (like a scrolling grid view) holding guards.
  // cell[i] displays idx = start + i.
  std::vector<size_t>                          cell_idx(window, static_cast<size_t>(-1));
  std::vector<std::shared_ptr<ThumbnailGuard>> cell_guard(window);

  size_t                                       drained = 0;
  size_t                                       center  = static_cast<size_t>(rng() % ids.size());
  size_t start   = (window >= ids.size()) ? 0 : std::min(center, max_start);

  auto   in_view = [&](size_t idx, size_t cur_start) -> bool {
    return idx >= cur_start && idx < (cur_start + window);
  };

  auto cell_pos_for = [&](size_t idx, size_t cur_start) -> size_t { return idx - cur_start; };

  auto request_idx  = [&](size_t idx, bool pin) {
    const auto element_id = ids[idx].first;
    const auto image_id   = ids[idx].second;

    auto       promise    = std::make_shared<std::promise<std::shared_ptr<ThumbnailGuard>>>();
    auto       fut        = promise->get_future();

    try {
      service.GetThumbnail(
          element_id, image_id,
          [promise, first_seen_guards, idx](std::shared_ptr<ThumbnailGuard> guard) {
            try {
              if (first_seen_guards && (*first_seen_guards)[idx].expired()) {
                (*first_seen_guards)[idx] = guard;
              }
              promise->set_value(guard);
            } catch (...) {
            }
          },
          pin);
      in_flight.push_back({idx, element_id, pin, std::move(fut)});
      in_flight_idx.insert(idx);
    } catch (std::exception& e) {
      // Swallow exceptions from GetThumbnail to keep fuzzing going.
      FAIL() << "GetThumbnail() threw exception for element ID " << element_id << ": " << e.what();
    }
  };

  auto wait_validate = [&](std::future<std::shared_ptr<ThumbnailGuard>> fut, size_t idx_for_debug) {
    std::shared_ptr<ThumbnailGuard> guard;
    WaitAndValidateFuture(std::move(fut), idx_for_debug, heavy_validate_every, drained, &guard);
    return guard;
  };

  auto drain_one = [&](size_t cur_start, bool force_block) {
    if (in_flight.empty()) {
      return false;
    }

    auto drain_item = [&](InFlight item) {
      in_flight_idx.erase(item.idx_);
      ++drained;
      auto guard = wait_validate(std::move(item.future_), item.idx_);
      if (!guard) {
        return true;
      }

      // If this idx is currently bound to a visible cell, attach it.
      if (in_view(item.idx_, cur_start)) {
        const size_t pos = cell_pos_for(item.idx_, cur_start);
        if (pos < window && cell_idx[pos] == item.idx_ && !cell_guard[pos]) {
          cell_guard[pos] = guard;
          return true;
        }
      }

      // Otherwise, behave like a UI that already scrolled away / never displayed it.
      EXPECT_NO_THROW(service.ReleaseThumbnail(item.element_id_));
      return true;
    };

    if (!force_block) {
      for (size_t i = 0; i < in_flight.size(); ++i) {
        if (in_flight[i].future_.wait_for(0s) == std::future_status::ready) {
          auto item = std::move(in_flight[i]);
          in_flight.erase(in_flight.begin() + static_cast<std::ptrdiff_t>(i));
          return drain_item(std::move(item));
        }
      }
      return false;
    }

    auto item = std::move(in_flight.front());
    in_flight.pop_front();
    return drain_item(std::move(item));
  };

  auto print_progress = [&](size_t iter) {
    if (!enable_progress || progress_every == 0) {
      return;
    }
    if ((iter % progress_every) != 0) {
      return;
    }
    const double pct  = (iterations == 0)
                            ? 100.0
                            : (100.0 * static_cast<double>(iter) / static_cast<double>(iterations));
    size_t       held = 0;
    for (const auto& g : cell_guard) {
      if (g) {
        ++held;
      }
    }
    std::cout << "\r\033[2K"
              << "[ThumbnailFuzzCells] iter=" << iter << "/" << iterations << " ("
              << static_cast<int>(pct) << "%)"
              << " drained=" << drained << " inflight=" << in_flight.size() << " held=" << held
              << std::flush;
  };

  for (size_t iter = 0; iter < iterations; ++iter) {
    print_progress(iter);

    // Drain ready work each iteration for throughput.
    while (drain_one(start, false)) {
    }

    int step = step_dist(rng);
    if (step == 0) {
      step = (pct_dist(rng) < 50) ? 1 : -1;
    }
    if (pct_dist(rng) < 4) {
      step *= static_cast<int>(window * 2);
    }

    center = reflect_center(static_cast<int>(center) + step);
    start  = (window >= ids.size())
                 ? 0
                 : std::clamp<size_t>((center > (window / 2)) ? (center - (window / 2)) : 0, 0,
                                     max_start);

    // Rebind each cell to the new viewport. Any old cell content is released immediately.
    for (size_t pos = 0; pos < window; ++pos) {
      const size_t want_idx = start + pos;
      if (cell_idx[pos] != want_idx) {
        if (cell_idx[pos] != static_cast<size_t>(-1) && cell_guard[pos]) {
          EXPECT_NO_THROW(service.ReleaseThumbnail(ids[cell_idx[pos]].first));
        }
        cell_idx[pos]   = want_idx;
        cell_guard[pos] = nullptr;
      }
    }

    auto maybe_request = [&](size_t idx, bool pin) {
      // If idx is visible and already attached to its cell, skip.
      if (in_view(idx, start)) {
        const size_t pos = cell_pos_for(idx, start);
        if (pos < window && cell_guard[pos]) {
          return;
        }
      }
      if (in_flight_idx.contains(idx)) {
        return;
      }
      request_idx(idx, pin);
      while (max_in_flight != 0 && in_flight.size() > max_in_flight) {
        if (drain_one(start, false)) {
          continue;
        }
        (void)drain_one(start, true);
      }
    };

    // Request thumbnails for visible cells first (pinned).
    for (size_t pos = 0; pos < window; ++pos) {
      maybe_request(start + pos, true);
    }

    // Prefetch around the viewport (not pinned).
    const size_t prefetch_begin = (start > prefetch_each_side) ? (start - prefetch_each_side) : 0;
    const size_t prefetch_end   = std::min(ids.size(), start + window + prefetch_each_side);
    for (size_t idx = prefetch_begin; idx < prefetch_end; ++idx) {
      if (in_view(idx, start)) {
        continue;
      }
      maybe_request(idx, false);
    }
  }

  while (drain_one(start, false)) {
  }
  while (!in_flight.empty()) {
    (void)drain_one(start, true);
  }

  // Simulate leaving the album view: release anything still shown in cells.
  for (size_t pos = 0; pos < window; ++pos) {
    if (cell_idx[pos] != static_cast<size_t>(-1) && cell_guard[pos]) {
      EXPECT_NO_THROW(service.ReleaseThumbnail(ids[cell_idx[pos]].first));
      cell_guard[pos] = nullptr;
    }
  }

  if (enable_progress) {
    std::cout << "\r\033[2K" << "[ThumbnailFuzzCells] iter=" << iterations << "/" << iterations
              << " (100%)"
              << " drained=" << drained << " inflight=0"
              << " held=0" << std::endl;
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

TEST_F(ThumbnailServiceTests, DISABLED_GenerateThumbnailAndCallbacks) {
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

TEST_F(ThumbnailServiceTests, DISABLED_PipelineRestoredFromDBGeneratesCorrectThumbnail) {
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

      auto pipeline  = pipeline_service->LoadPipeline(file_id);
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
      pipline_after             = pipeline->pipeline_->ExportPipelineParams().dump();
      ASSERT_NE(pipline_before, pipline_after)
          << "Pipeline parameters did not change after modification";
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

TEST_F(ThumbnailServiceTests, DISABLED_FuzzScrollBrowsingNoThrowReloadService) {
  // Simulate a user scrolling back and forth in the UI thumbnail grid.
  // Requirement: no throws; two phases; and on each "service shutdown" persist like
  // PipelineRestoredFromDBGeneratesCorrectThumbnail.

  // Phase 0: import images once and persist the project.
  std::vector<std::pair<sl_element_id_t, image_id_t>> ids;
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
      FuzzScrollRequestsNoThrow(thumbnail_service, ids, 5000, 0xC0FFEEu);
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
      FuzzScrollRequestsNoThrow(thumbnail_service, ids, 5000, 0xBADC0DEu);
      ReleaseAllThumbnailsAggressively(thumbnail_service, ids);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }
}

TEST_F(ThumbnailServiceTests, FuzzScrollBrowsingSharedPtrLifetimeStress) {
  // Goal: run many scroll-like iterations without accumulating unbounded futures,
  // and ensure ThumbnailGuard shared_ptrs do not outlive the ThumbnailService.
  // Pattern requirement: two phases; and on each "service shutdown" persist like
  // PipelineRestoredFromDBGeneratesCorrectThumbnail.

  constexpr size_t                                    kIterations = 50'000;

  std::vector<std::pair<sl_element_id_t, image_id_t>> ids;

  // Phase 0: import images once and persist.
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
    ASSERT_GE(paths.size(), 16u) << "Need images under TEST_IMG_PATH/raw/batch_import";

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

    std::cout << "[ThumbnailFuzz] Imported " << paths.size() << " images for fuzzing." << std::endl;

    const auto final_result_value = final_result_future.get();
    ASSERT_EQ(final_result_value.failed_, 0u);
    ASSERT_NE(import_job->import_log_, nullptr);
    auto snapshot = import_job->import_log_->Snapshot();
    ASSERT_FALSE(snapshot.created_.empty());

    const size_t count = std::min<size_t>(128, snapshot.created_.size());
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      ids.push_back({snapshot.created_[i].element_id_, snapshot.created_[i].image_id_});
    }

    import_service.SyncImports(snapshot, L"");
    std::cout << "[ThumbnailFuzz] Synced imports to storage." << std::endl;

    project.GetSleeveService()->Sync();
    project.GetImagePoolService()->SyncWithStorage();
    project.SaveProject(meta_path_);
    std::cout << "[ThumbnailFuzz] Project saved." << std::endl;
  }

  ASSERT_FALSE(ids.empty());
  std::vector<std::weak_ptr<ThumbnailGuard>> phase1_weak(ids.size());
  std::vector<std::weak_ptr<ThumbnailGuard>> phase2_weak(ids.size());

  // Phase 1: stress browse.
  std::cout << "[ThumbnailFuzz] Starting phase 1 browsing fuzz..." << std::endl;
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

      // // Ensure each id is observed at least once, then release.
      // for (size_t i = 0; i < ids.size(); ++i) {
      //   const auto [id, image_id] = ids[i];
      //   auto       guard          = GetThumbnailBlocking(thumbnail_service, id, image_id, true);
      //   phase1_weak[i]            = guard;
      //   thumbnail_service.ReleaseThumbnail(id);
      // }

      // Simulate real album scrolling behavior:
      // - A fixed 50-cell grid; each scroll step rebinds cells.
      // - When a cell scrolls off-screen, the thumbnail is released immediately.
      // - Prefetch around the viewport to match cache behavior.
      // - Bound outstanding requests (in-flight) to maximize throughput without overload.
      FuzzAlbumScrollGridCellsNoThrow(thumbnail_service, ids, kIterations, 0xFEEDFACEu,
                                      /*view_size=*/50,
                                      /*prefetch_each_side=*/7,
                                      /*max_in_flight=*/12,
                                      /*heavy_validate_every=*/5000, &phase1_weak,
                                      /*enable_progress=*/true,
                                      /*progress_every=*/50);
      ReleaseAllThumbnailsAggressively(thumbnail_service, ids, 64);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }

  for (size_t i = 0; i < phase1_weak.size(); ++i) {
    EXPECT_TRUE(phase1_weak[i].expired())
        << "ThumbnailGuard leaked after service shutdown (idx=" << i << ")";
  }

  // Phase 2: reload service and stress again.
  {
    ProjectService project(db_path_, meta_path_);
    auto           img_pool = project.GetImagePoolService();
    auto pipeline_service   = std::make_shared<PipelineMgmtService>(project.GetStorageService());

    {
      ThumbnailService thumbnail_service(project.GetSleeveService(), img_pool, pipeline_service);

      FuzzAlbumScrollGridCellsNoThrow(thumbnail_service, ids, kIterations, 0x1234ABCDu,
                                      /*view_size=*/50,
                                      /*prefetch_each_side=*/7,
                                      /*max_in_flight=*/12,
                                      /*heavy_validate_every=*/5000, &phase2_weak,
                                      /*enable_progress=*/true,
                                      /*progress_every=*/50);
      ReleaseAllThumbnailsAggressively(thumbnail_service, ids, 64);
    }

    pipeline_service->Sync();
    img_pool->SyncWithStorage();
    project.SaveProject(meta_path_);
  }

  for (size_t i = 0; i < phase2_weak.size(); ++i) {
    EXPECT_TRUE(phase2_weak[i].expired())
        << "ThumbnailGuard leaked after service reload (idx=" << i << ")";
  }
}

TEST_F(ThumbnailServiceTests, DISABLED_Generate16ThumbnailsAndValidateAll) {
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
        id, image_id,
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
