//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

// A project written by an earlier build opens and renders unchanged
// (raster_image_input_plan.md, section 7.4).
//
// tests/resources/compat/raw_project_abdb000c8.alcd was packed by build abdb000c8 (main before
// raster input). It holds one CI RAW image with one exposure commit. The JSON beside it records
// the committed document and the render hash of support/project_compat_render.hpp on that
// build. The hash is for CUDA on the GPU that recorded it, so only a CUDA build compares it;
// every build checks that the project opens with the same document and renders.

#include <duckdb.h>
#include <gtest/gtest.h>

#include <QString>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>

#include "app/pipeline_service.hpp"
#include "app/project_package_backend.hpp"
#include "app/project_service.hpp"
#include "edit/history/commit_graph.hpp"
#include "edit/history/edit_commit_attribution.hpp"
#include "edit/history/mini_git_working_history.hpp"
#include "edit/history/pipeline_edit_batch.hpp"
#include "json.hpp"
#include "storage/store/edit_history/commit_graph_store.hpp"
#include "support/project_compat_render.hpp"
#include "utils/clock/time_provider.hpp"
#include "utils/string/convert.hpp"

namespace alcedo {
namespace {

auto CompatDir() -> std::filesystem::path { return ALCEDO_COMPAT_FIXTURE_DIR; }

/// The fixture stores the RAW path of the machine that wrote it. Point it at this checkout.
void RewriteImagePath(const std::filesystem::path& db_path, const std::filesystem::path& raw_path) {
  duckdb_database   database;
  duckdb_connection connection;
  ASSERT_EQ(duckdb_open(db_path.string().c_str(), &database), DuckDBSuccess);
  ASSERT_EQ(duckdb_connect(database, &connection), DuckDBSuccess);
  duckdb_prepared_statement statement;
  ASSERT_EQ(duckdb_prepare(connection, "UPDATE Image SET image_path = ?", &statement),
            DuckDBSuccess);
  const auto path_bytes = conv::ToBytes(raw_path.wstring());
  ASSERT_EQ(duckdb_bind_varchar(statement, 1, path_bytes.c_str()), DuckDBSuccess);
  duckdb_result result;
  ASSERT_EQ(duckdb_execute_prepared(statement, &result), DuckDBSuccess);
  EXPECT_EQ(duckdb_rows_changed(&result), 1u);
  duckdb_destroy_result(&result);
  duckdb_destroy_prepare(&statement);
  duckdb_disconnect(&connection);
  duckdb_close(&database);
}

TEST(ProjectCompatibilityTest, ProjectWrittenByCurrentMainOpensAndRendersUnchanged) {
  TimeProvider::Refresh();
  nlohmann::json expected;
  {
    std::ifstream in(CompatDir() / "raw_project_abdb000c8.json", std::ios::binary);
    ASSERT_TRUE(in.is_open());
    expected = nlohmann::json::parse(in);
  }
  const auto raw_path = std::filesystem::path(TEST_IMG_PATH) / "ci_rawfiles" /
                        expected.at("raw_fixture").get<std::string>();
  if (!std::filesystem::exists(raw_path)) {
    GTEST_SKIP() << "CI RAW fixture is missing (Git LFS): " << raw_path.string();
  }

  const auto      workspace = std::filesystem::temp_directory_path() / "alcedo_project_compat";
  std::error_code ec;
  std::filesystem::remove_all(workspace, ec);
  std::filesystem::path db_path;
  std::filesystem::path meta_path;
  QString               error;
  ASSERT_TRUE(project_pack::UnpackProjectToWorkspace(CompatDir() / "raw_project_abdb000c8.alcd",
                                                     workspace, QStringLiteral("Compat"), &db_path,
                                                     &meta_path, &error))
      << error.toStdString();
  RewriteImagePath(db_path, raw_path);

  const auto element_id = expected.at("element_id").get<sl_element_id_t>();
  const auto image_id   = expected.at("image_id").get<image_id_t>();
  {
    ProjectService      project(db_path, meta_path);
    PipelineMgmtService pipelines(project.GetStorage());
    const auto          snapshot = pipelines.AcquireCommittedSnapshot(element_id);
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->Document().ToJson(), expected.at("committed_document"));
    EXPECT_FALSE(snapshot->Document().Develop()->Params().RasterInput().has_value());

    const auto hash = test::RenderCommittedElementHash(pipelines, *project.GetImagePoolService(),
                                                       element_id, image_id);
    EXPECT_NE(hash, 0u);
#ifdef HAVE_CUDA
    // The recorded hash is a CUDA render; other backends round differently.
    EXPECT_EQ(std::to_string(hash), expected.at("render_hash").get<std::string>());
#endif
  }
  std::filesystem::remove_all(workspace, ec);
}

TEST(ProjectCompatibilityTest, OldProjectCommitsReadAsUserAndSaveAnAgentCommit) {
  TimeProvider::Refresh();
  nlohmann::json expected;
  {
    std::ifstream in(CompatDir() / "raw_project_abdb000c8.json", std::ios::binary);
    ASSERT_TRUE(in.is_open());
    expected = nlohmann::json::parse(in);
  }
  const auto workspace =
      std::filesystem::temp_directory_path() / "alcedo_project_compat_attribution";
  std::error_code ec;
  std::filesystem::remove_all(workspace, ec);
  std::filesystem::path db_path;
  std::filesystem::path meta_path;
  QString               error;
  ASSERT_TRUE(project_pack::UnpackProjectToWorkspace(CompatDir() / "raw_project_abdb000c8.alcd",
                                                     workspace, QStringLiteral("Compat"), &db_path,
                                                     &meta_path, &error))
      << error.toStdString();
  const auto element_id = expected.at("element_id").get<sl_element_id_t>();

  SetParameterChange change;
  change.target.owner_kind             = PipelineParameterOwnerKind::ColorGrade;
  change.target.node_id                = NodeId{"grade.primary"};
  change.target.adjustment_instance_id = AdjustmentInstanceId{"grade.primary.contrast"};
  change.target.field_key              = "contrast";
  change.before_value                  = nlohmann::json{{"contrast", 0.0}};
  change.after_value                   = nlohmann::json{{"contrast", 12.0}};
  change.before_enabled                = true;
  change.after_enabled                 = true;
  auto batch = PipelineEditBatch::Make(PipelineEditOperationKind::SetParameter, {change},
                                       "history.operation.set_parameter");

  commit_hash_t agent_hash{};
  std::size_t   commit_count = 0;
  {
    ProjectService   project(db_path, meta_path);
    auto             guard = project.GetStorage()->GetDatabase().GetConnectionGuard();
    auto             lock  = guard.Lock();
    CommitGraphStore store(guard.conn_);
    auto             loaded = store.LoadGraph(element_id);
    ASSERT_TRUE(loaded.has_value());
    ASSERT_GT(loaded->CommitCount(), 0u);
    EXPECT_TRUE(loaded->GetAllCommitAttributions().empty())
        << "the commits of an old project are the user's";

    auto                  graph = std::make_shared<CommitGraph>(std::move(*loaded));
    MiniGitWorkingHistory history(graph, std::make_shared<MiniGitJournal>());
    const auto            prepared = history.PrepareAppendEdit(
        std::move(batch),
        EditCommitAttribution{std::string(kEditCommitAuthorAgent), "compat edit"});
    ASSERT_TRUE(prepared.ready) << prepared.error;
    const auto appended = history.PublishPreparedEdit(prepared);
    ASSERT_TRUE(appended.committed) << appended.error;
    agent_hash   = appended.commit->GetCommitHash();
    commit_count = graph->CommitCount();
    ASSERT_NO_THROW(store.Materialize(graph->CaptureMaterialization()));
  }
  {
    ProjectService   project(db_path, meta_path);
    auto             guard = project.GetStorage()->GetDatabase().GetConnectionGuard();
    auto             lock  = guard.Lock();
    CommitGraphStore store(guard.conn_);
    const auto       reloaded = store.LoadGraph(element_id);
    ASSERT_TRUE(reloaded.has_value());
    EXPECT_EQ(reloaded->CommitCount(), commit_count);
    EXPECT_EQ(reloaded->GetAllCommitAttributions().size(), 1u);
    const auto* attribution = reloaded->FindCommitAttribution(agent_hash);
    ASSERT_NE(attribution, nullptr);
    EXPECT_EQ(attribution->author_, kEditCommitAuthorAgent);
    EXPECT_EQ(attribution->note_, "compat edit");
  }
  std::filesystem::remove_all(workspace, ec);
}

}  // namespace
}  // namespace alcedo
