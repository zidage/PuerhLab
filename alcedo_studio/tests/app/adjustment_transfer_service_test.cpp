//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/adjustment_transfer_service.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

#include "edit/graph/drt_node_model.hpp"
#include "edit/graph/pipeline_document.hpp"
#include "edit/history/commit_graph.hpp"
#include "edit/history/commit_types.hpp"
#include "edit/history/edit_commit.hpp"
#include "edit/history/edit_commit_attribution.hpp"
#include "edit/history/pipeline_edit_batch.hpp"
#include "edit/history/pipeline_history_format.hpp"
#include "json.hpp"
#include "support/document_transfer_test_support.hpp"
#include "support/editor_mini_git_project_fixture.hpp"

namespace alcedo {
namespace {

TEST(AdjustmentTransferServiceTest, CaptureExportImportRoundTripKeepsDevelopAndGeometry) {
  auto source = test::DocumentWithExposureEv(1.5);
  source.Geometry().SetRotationDegrees(12.0f);
  const auto package  = AdjustmentTransferService::Capture(source);
  const auto exported = AdjustmentTransferService::ExportPackage(package);
  EXPECT_FALSE(exported.contains("operators"));
  ASSERT_TRUE(exported.contains("develop"));
  EXPECT_FLOAT_EQ(exported.at("develop").at("geometry").at("rotation_degrees").get<float>(), 12.0f);
  EXPECT_FALSE(exported.at("develop").at("raw_decode").contains("camera_profile"));
  const auto imported = AdjustmentTransferService::ImportPackage(exported);
  EXPECT_EQ(imported.fingerprint_, package.fingerprint_);
  EXPECT_FALSE(imported.Empty());
}

TEST(AdjustmentTransferServiceTest, ImportRejectsOperatorListPackages) {
  const nlohmann::json json = {{"operators", nlohmann::json::array({nlohmann::json::object()})},
                               {"schema", std::string{kAdjustmentTransferSchema}}};
  EXPECT_THROW((void)AdjustmentTransferService::ImportPackage(json), std::runtime_error);
}

TEST(AdjustmentTransferServiceTest, TransferSurfaceHasNoPipelineMergeOperation) {
  test::EditorMiniGitProjectFixture project;
  project.SetUp();
  auto* graph = project.graph(test::EditorMiniGitProjectFixture::kElementA).get();
  ASSERT_NE(graph, nullptr);
  const auto package = test::MakeDefaultDocumentTransferPackage();
  const auto pasted  = AdjustmentTransferService::PasteAsRootRelativeVersion(
      *graph, CreateDefaultPipelineDocument(), package, "Pasted Adjustments");
  ASSERT_TRUE(pasted.pasted) << pasted.error;
  EXPECT_EQ(graph->CommitCount(), 1u);
  const auto& commit = graph->GetCommit(pasted.new_head);
  ASSERT_TRUE(IsPipelineEditBatchJson(commit.GetPayloadJSON()));
  EXPECT_EQ(PipelineEditBatch::FromJSON(commit.GetPayloadJSON()).operation_kind,
            PipelineEditOperationKind::Paste);
  // A Paste without attribution is a commit of the user.
  EXPECT_EQ(graph->FindCommitAttribution(pasted.new_head), nullptr);
  project.TearDown();
}

TEST(AdjustmentTransferServiceTest, AgentPasteCommitCarriesAgentAttribution) {
  test::EditorMiniGitProjectFixture project;
  project.SetUp();
  auto* graph = project.graph(test::EditorMiniGitProjectFixture::kElementA).get();
  ASSERT_NE(graph, nullptr);
  const EditCommitAttribution attribution{std::string(kEditCommitAuthorAgent), "apply the look"};
  const auto                  pasted = AdjustmentTransferService::PasteAsRootRelativeVersion(
      *graph, CreateDefaultPipelineDocument(), test::MakeDefaultDocumentTransferPackage(),
      "Pasted Adjustments", {}, attribution);
  ASSERT_TRUE(pasted.pasted) << pasted.error;
  const auto* stored = graph->FindCommitAttribution(pasted.new_head);
  ASSERT_NE(stored, nullptr);
  EXPECT_EQ(*stored, attribution);
  project.TearDown();
}

TEST(AdjustmentTransferServiceTest, PasteWithUserAuthorAttributionFailsAndKeepsActiveVersion) {
  test::EditorMiniGitProjectFixture project;
  project.SetUp();
  auto* graph = project.graph(test::EditorMiniGitProjectFixture::kElementA).get();
  ASSERT_NE(graph, nullptr);
  const auto prior_version = graph->GetActiveVersionId();
  const auto prior_count   = graph->GetAllVersionRefs().size();
  const auto pasted        = AdjustmentTransferService::PasteAsRootRelativeVersion(
      *graph, CreateDefaultPipelineDocument(), test::MakeDefaultDocumentTransferPackage(),
      "Pasted Adjustments", {},
      EditCommitAttribution{std::string(kEditCommitAuthorUser), std::string()});
  EXPECT_FALSE(pasted.pasted);
  EXPECT_FALSE(pasted.error.empty());
  EXPECT_EQ(graph->GetActiveVersionId(), prior_version);
  EXPECT_EQ(graph->GetAllVersionRefs().size(), prior_count);
  project.TearDown();
}

// The transfer coordinator persists the library HDR flag from IsHdrExportEncoding(document DRT).
TEST(AdjustmentTransferServiceTest, HdrExportFlagFollowsDocumentDrtEncoding) {
  auto  document = CreateDefaultPipelineDocument();
  auto* drt      = document.Drt();
  ASSERT_NE(drt, nullptr);
  EXPECT_FALSE(IsHdrExportEncoding(*drt));

  DrtParameterUpdate pq;
  pq.encoding_eotf = DrtEotf::St2084;
  drt->Params().ApplyUpdate(pq);
  EXPECT_TRUE(IsHdrExportEncoding(*drt));

  DrtParameterUpdate hlg;
  hlg.encoding_eotf = DrtEotf::Hlg;
  drt->Params().ApplyUpdate(hlg);
  EXPECT_TRUE(IsHdrExportEncoding(*drt));

  DrtParameterUpdate gamma;
  gamma.encoding_eotf = DrtEotf::Gamma22;
  drt->Params().ApplyUpdate(gamma);
  EXPECT_FALSE(IsHdrExportEncoding(*drt));
  EXPECT_EQ(drt->Params().ToJson(), CreateDefaultPipelineDocument().Drt()->Params().ToJson());
}

}  // namespace
}  // namespace alcedo
