#include "app/pipeline_service.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <atomic>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "app/project_service.hpp"
#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_registeration.hpp"
#include "sleeve/storage_service.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {

class PipelineServiceTests : public ::testing::Test {
 protected:
  std::filesystem::path db_path_;
  std::filesystem::path meta_path_;

  void                  SetUp() override {
    TimeProvider::Refresh();
    db_path_ = std::filesystem::temp_directory_path() / "sleeve_service_test.db";
    meta_path_ = std::filesystem::temp_directory_path() / "sleeve_service_test.json";
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
    RegisterAllOperators();
  }

  void TearDown() override {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove(db_path_);
    }
    if (std::filesystem::exists(meta_path_)) {
      std::filesystem::remove(meta_path_);
    }
  }
};

TEST_F(PipelineServiceTests, InitTest) {
  ProjectService project(db_path_, meta_path_);
  EXPECT_NO_THROW(PipelineMgmtService pipeline_service(project.GetStorageService()));
}

TEST_F(PipelineServiceTests, BasicPipelineRWTest) {
  std::string pipeline_param;
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    // Load a pipeline that does not exist yet, should get a new pipeline
    auto                pipeline_guard = pipeline_service.LoadPipeline(1);

    EXPECT_NE(pipeline_guard, nullptr);
    EXPECT_EQ(pipeline_guard->id_, 1);
    EXPECT_EQ(pipeline_guard->pinned_, true);
    EXPECT_EQ(pipeline_guard->dirty_, false);

    // Modify the pipeline
    auto&          stage = pipeline_guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
    nlohmann::json exp_params;
    exp_params["exposure"] = 1.5f;
    stage.SetOperator(OperatorType::EXPOSURE, exp_params);
    pipeline_guard->dirty_ = true;

    // Save it back
    pipeline_service.SavePipeline(pipeline_guard);

    // Since SavePipeline() will only write back to the cache, we need to call Sync() to write to DB
    pipeline_service.Sync();

    // Load it again and serialize the pipeline to compare
    auto pipeline_guard_2 = pipeline_service.LoadPipeline(1);
    EXPECT_NE(pipeline_guard_2, nullptr);
    EXPECT_EQ(pipeline_guard_2->id_, 1);
    EXPECT_EQ(pipeline_guard_2->pinned_, true);
    EXPECT_EQ(pipeline_guard_2->dirty_,
              false);  // We have sync the cache, so it should not be dirty
    // Serialize it
    pipeline_param = pipeline_guard_2->pipeline_->ExportPipelineParams().dump(2);
  }
  // Leave the scope, reopen and load again
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    auto                pipeline_guard = pipeline_service.LoadPipeline(1);
    EXPECT_NE(pipeline_guard, nullptr);
    EXPECT_EQ(pipeline_guard->id_, 1);
    EXPECT_EQ(pipeline_guard->pinned_, true);
    EXPECT_EQ(pipeline_guard->dirty_, false);  // Not dirty since we just loaded it
    // Serialize it
    auto pipeline_param_2 = pipeline_guard->pipeline_->ExportPipelineParams().dump(2);
    EXPECT_EQ(pipeline_param, pipeline_param_2);
  }
}

TEST_F(PipelineServiceTests, SharedGuardPinsUntilLastSave) {
  ProjectService      project(db_path_, meta_path_);
  PipelineMgmtService pipeline_service(project.GetStorageService());

  auto                guard_a = pipeline_service.LoadPipeline(7);
  auto                guard_b = pipeline_service.LoadPipeline(7);

  ASSERT_NE(guard_a, nullptr);
  ASSERT_NE(guard_b, nullptr);
  EXPECT_EQ(guard_a.get(), guard_b.get());
  EXPECT_TRUE(guard_a->pinned_);
  EXPECT_EQ(guard_a->pin_count_, 2u);

  pipeline_service.SavePipeline(guard_a);
  EXPECT_TRUE(guard_b->pinned_);
  EXPECT_EQ(guard_b->pin_count_, 1u);

  pipeline_service.SavePipeline(guard_b);
  EXPECT_FALSE(guard_b->pinned_);
  EXPECT_EQ(guard_b->pin_count_, 0u);

  auto guard_c = pipeline_service.LoadPipeline(7);
  ASSERT_NE(guard_c, nullptr);
  EXPECT_TRUE(guard_c->pinned_);
  EXPECT_EQ(guard_c->pin_count_, 1u);
}

TEST_F(PipelineServiceTests, MultiplePipelineTest) {
  constexpr int                           pipeline_count = 5;
  std::array<std::string, pipeline_count> pipeline_params;
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    // Create and save multiple pipelines
    for (sl_element_id_t i = 1; i <= pipeline_count; ++i) {
      auto pipeline_guard = pipeline_service.LoadPipeline(i);
      EXPECT_NE(pipeline_guard, nullptr);
      EXPECT_EQ(pipeline_guard->id_, i);

      // Modify the pipeline
      auto& stage = pipeline_guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
      nlohmann::json exp_params;
      exp_params["contrast"] = static_cast<float>(i) * 0.5f;
      stage.SetOperator(OperatorType::CONTRAST, exp_params);
      pipeline_guard->dirty_ = true;

      // Save it back
      pipeline_service.SavePipeline(pipeline_guard);
      pipeline_params[i - 1] = pipeline_guard->pipeline_->ExportPipelineParams().dump(2);
    }
    // Sync to DB
    pipeline_service.Sync();
  }

  // Reopen and load again to verify
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    for (sl_element_id_t i = 1; i <= pipeline_count; ++i) {
      auto pipeline_guard = pipeline_service.LoadPipeline(i);
      EXPECT_NE(pipeline_guard, nullptr);
      EXPECT_EQ(pipeline_guard->id_, i);

      // Serialize it
      auto pipeline_param_2 = pipeline_guard->pipeline_->ExportPipelineParams().dump(2);
      EXPECT_EQ(pipeline_params[i - 1], pipeline_param_2);
    }
  }
}

TEST_F(PipelineServiceTests, CacheTest1) {
  {
    ProjectService                              project(db_path_, meta_path_);
    PipelineMgmtService                         pipeline_service(project.GetStorageService());

    // The default cache size is 64, so we will create 65 pipelines to exceed the cache size
    constexpr int                               pipeline_count = 65;
    std::array<sl_element_id_t, pipeline_count> pipeline_ids;
    for (sl_element_id_t i = 1; i <= pipeline_count; ++i) {
      auto pipeline_guard = pipeline_service.LoadPipeline(i);
      EXPECT_NE(pipeline_guard, nullptr);
      EXPECT_EQ(pipeline_guard->id_, i);
      pipeline_ids[i - 1] = i;

      // Modify the pipeline
      auto& stage         = pipeline_guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
      nlohmann::json exp_params;
      exp_params["exposure"] = static_cast<float>(i) * 0.3f;
      stage.SetOperator(OperatorType::EXPOSURE, exp_params);
      pipeline_guard->dirty_ = true;
      // Save it back
      // So no guard will be pinned
      pipeline_service.SavePipeline(pipeline_guard);
    }
    // Now try to access the first pipeline again, it should be evicted and synced to DB, so it is
    // not dirty
    auto first_pipeline_guard = pipeline_service.LoadPipeline(pipeline_ids[0]);
    EXPECT_NE(first_pipeline_guard, nullptr);
    EXPECT_EQ(first_pipeline_guard->id_, pipeline_ids[0]);
    EXPECT_EQ(first_pipeline_guard->dirty_, false);
  }
}

TEST_F(PipelineServiceTests, CacheTest2) {
  {
    ProjectService                              project(db_path_, meta_path_);
    PipelineMgmtService                         pipeline_service(project.GetStorageService());

    // The default cache size is 64, so we will create 70 pipelines to exceed the cache size
    constexpr int                               pipeline_count = 70;
    std::array<sl_element_id_t, pipeline_count> pipeline_ids;
    for (sl_element_id_t i = 0; i < pipeline_count; ++i) {
      auto pipeline_guard = pipeline_service.LoadPipeline(i);
      EXPECT_NE(pipeline_guard, nullptr);
      EXPECT_EQ(pipeline_guard->id_, i);
      pipeline_ids[i] = i;

      // Modify the pipeline
      auto& stage     = pipeline_guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
      nlohmann::json exp_params;
      exp_params["contrast"] = static_cast<float>(i) * 0.4f;
      stage.SetOperator(OperatorType::CONTRAST, exp_params);
      pipeline_guard->dirty_ = true;

      // No save back, so all pipelines are in use
    }
    // Now try to access the first pipeline again, it should still be in the cache and dirty
    auto first_pipeline_guard = pipeline_service.LoadPipeline(pipeline_ids[0]);
    EXPECT_NE(first_pipeline_guard, nullptr);
    EXPECT_EQ(first_pipeline_guard->id_, pipeline_ids[0]);
    EXPECT_EQ(first_pipeline_guard->dirty_, true);
  }
}

TEST_F(PipelineServiceTests, DISABLED_FuzzTest) {
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    constexpr int                 kOpsCount        = 500;
    constexpr int                 kIdRange         = 96;
    std::mt19937                  rng{12345};
    std::uniform_int_distribution<int> id_dist(1, kIdRange);
    std::uniform_int_distribution<int> op_dist(0, 5);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);
    std::unordered_map<sl_element_id_t, std::string> expected_dump;
    const auto empty_dump = CPUPipelineExecutor().ExportPipelineParams().dump();

    for (int i = 0; i < kOpsCount; ++i) {
      const auto id = static_cast<sl_element_id_t>(id_dist(rng));
      const auto op = op_dist(rng);

      if (op == 0) {
        // Load pipeline (cache hit/miss paths)
        auto guard = pipeline_service.LoadPipeline(id);
        ASSERT_NE(guard, nullptr);
        EXPECT_EQ(guard->id_, id);
        auto dump = guard->pipeline_->ExportPipelineParams().dump();
        if (expected_dump.contains(id)) {
          EXPECT_EQ(dump, expected_dump.at(id));
        } else {
          // If we never wrote an ID-bound param, it should still be empty
          EXPECT_EQ(dump, empty_dump);
        }
      } else if (op == 1) {
        // Load + modify + save (dirty path)
        auto guard = pipeline_service.LoadPipeline(id);
        ASSERT_NE(guard, nullptr);
        auto& stage = guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
        nlohmann::json params;
        params["exposure"] = static_cast<float>(id) + value_dist(rng);
        stage.SetOperator(OperatorType::EXPOSURE, params);
        guard->dirty_ = true;
        pipeline_service.SavePipeline(guard);
        expected_dump[id] = guard->pipeline_->ExportPipelineParams().dump();
      } else if (op == 2) {
        // Load + modify without save (pinned & dirty in cache)
        auto guard = pipeline_service.LoadPipeline(id);
        ASSERT_NE(guard, nullptr);
        auto& stage = guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
        nlohmann::json params;
        params["contrast"] = static_cast<float>(id) + value_dist(rng);
        stage.SetOperator(OperatorType::CONTRAST, params);
        guard->dirty_ = true;
        expected_dump[id] = guard->pipeline_->ExportPipelineParams().dump();
      } else if (op == 3) {
        // Sync all dirty pipelines
        pipeline_service.Sync();
      } else if (op == 4) {
        // Stress eviction by accessing a far ID
        auto guard = pipeline_service.LoadPipeline(static_cast<sl_element_id_t>(kIdRange + id));
        ASSERT_NE(guard, nullptr);
        EXPECT_EQ(guard->id_, static_cast<sl_element_id_t>(kIdRange + id));
        auto dump = guard->pipeline_->ExportPipelineParams().dump();
        const auto far_id = static_cast<sl_element_id_t>(kIdRange + id);
        if (expected_dump.contains(far_id)) {
          EXPECT_EQ(dump, expected_dump.at(far_id));
        } else {
          EXPECT_EQ(dump, empty_dump);
        }
      } else {
        // Random read/serialize path
        auto guard = pipeline_service.LoadPipeline(id);
        ASSERT_NE(guard, nullptr);
        auto serialized = guard->pipeline_->ExportPipelineParams().dump();
        if (expected_dump.contains(id)) {
          EXPECT_EQ(serialized, expected_dump.at(id));
        } else {
          EXPECT_EQ(serialized, empty_dump);
        }
      }
    }

    pipeline_service.Sync();
  }

  // Reopen to verify some pipelines persisted and can be read
  {
    ProjectService      project(db_path_, meta_path_);
    PipelineMgmtService pipeline_service(project.GetStorageService());

    for (sl_element_id_t id = 1; id <= 10; ++id) {
      auto guard = pipeline_service.LoadPipeline(id);
      ASSERT_NE(guard, nullptr);
      EXPECT_EQ(guard->id_, id);
      auto serialized = guard->pipeline_->ExportPipelineParams().dump();
      EXPECT_FALSE(serialized.empty());
    }
  }
}

TEST_F(PipelineServiceTests, DISABLED_ThreadSafeTest) {
  ProjectService      project(db_path_, meta_path_);
  PipelineMgmtService pipeline_service(project.GetStorageService());

  constexpr int kThreads   = 8;
  constexpr int kOpsPerThr = 200;
  constexpr int kIdRange   = 64;

  std::atomic<int> ops_count{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([t, &pipeline_service, &ops_count]() {
      for (int i = 0; i < kOpsPerThr; ++i) {
        const auto id = static_cast<sl_element_id_t>((t * kOpsPerThr + i) % kIdRange + 1);
        auto       guard = pipeline_service.LoadPipeline(id);
        ASSERT_NE(guard, nullptr);
        auto& stage = guard->pipeline_->GetStage(PipelineStageName::To_WorkingSpace);
        nlohmann::json params;
        params["exposure"] = static_cast<float>(id) + static_cast<float>(t) * 0.01f;
        stage.SetOperator(OperatorType::EXPOSURE, params);
        guard->dirty_ = true;
        pipeline_service.SavePipeline(guard);
        if (i % 10 == 0) {
          pipeline_service.Sync();
        }
        ++ops_count;
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  pipeline_service.Sync();
  EXPECT_EQ(ops_count.load(), kThreads * kOpsPerThr);

  const auto empty_dump = CPUPipelineExecutor().ExportPipelineParams().dump();
  for (sl_element_id_t id = 1; id <= 10; ++id) {
    auto guard = pipeline_service.LoadPipeline(id);
    ASSERT_NE(guard, nullptr);
    auto serialized = guard->pipeline_->ExportPipelineParams().dump();
    EXPECT_NE(serialized, empty_dump);
  }
}
}  // namespace puerhlab
