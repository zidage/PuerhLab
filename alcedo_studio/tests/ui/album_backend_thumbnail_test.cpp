//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/album_backend_test_fixture.hpp"

#include <QSignalSpy>

#include <filesystem>
#include <functional>

#include "app/pipeline_service.hpp"
#include "app/project_service.hpp"
#include "edit/pipeline/default_pipeline_params.hpp"
#ifdef HAVE_METAL
#include "image/metal_image.hpp"
#endif

namespace alcedo::ui::test {
namespace {

using ThumbnailTests = AlbumBackendTestFixture;

auto MetalAvailable() -> bool {
#ifdef HAVE_METAL
  auto* device = MTL::CreateSystemDefaultDevice();
  if (device == nullptr) {
    return false;
  }
  device->release();
  return true;
#else
  return false;
#endif
}

void WaitForImportFinished(AlbumBackend& backend, int timeoutMs = 30000) {
  QSignalSpy spy(&backend, &AlbumBackend::ImportStateChanged);
  const int  step_ms = 200;
  int        elapsed = 0;
  while (backend.ImportRunning() && elapsed < timeoutMs) {
    spy.wait(step_ms);
    elapsed += step_ms;
  }
  ProcessEvents(500);
}

auto WaitForThumbnailUrl(AlbumBackend& backend, sl_element_id_t element_id,
                         bool expect_non_empty, int timeout_ms = 30000) -> QString {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    const QVariantList thumbnails = backend.Thumbnails();
    for (const QVariant& row_value : thumbnails) {
      const QVariantMap row = row_value.toMap();
      if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != element_id) {
        continue;
      }
      const QString thumb_url = row.value("thumbUrl").toString();
      if (expect_non_empty ? !thumb_url.isEmpty() : thumb_url.isEmpty()) {
        return thumb_url;
      }
      break;
    }
    ProcessEvents(100);
  }
  return {};
}

auto WaitForThumbnailRow(AlbumBackend& backend, sl_element_id_t element_id,
                         const std::function<bool(const QVariantMap&)>& predicate,
                         int timeout_ms = 30000) -> QVariantMap {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    const QVariantList thumbnails = backend.Thumbnails();
    for (const QVariant& row_value : thumbnails) {
      const QVariantMap row = row_value.toMap();
      if (static_cast<sl_element_id_t>(row.value("elementId").toUInt()) != element_id) {
        continue;
      }
      if (predicate(row)) {
        return row;
      }
      break;
    }
    ProcessEvents(100);
  }
  return {};
}

}  // namespace

TEST_F(ThumbnailTests, MetalThumbnailGridLifecycleWithGeometryOperatorsProducesDataUrl) {
#ifndef HAVE_METAL
  GTEST_SKIP() << "Metal is not enabled in this build.";
#else
  if (!MetalAvailable()) {
    GTEST_SKIP() << "Metal device is unavailable in this environment.";
  }

  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("still_life", 1);
  if (images.empty()) {
    images = CollectRawTestImages("airplane", 1);
  }
  ASSERT_FALSE(images.empty()) << "No RAW test image available for thumbnail regression.";

  backend.StartImport(PathsToQStringList(images));
  WaitForImportFinished(backend);

  ASSERT_FALSE(backend.ImportRunning());
  ASSERT_GE(backend.ShownCount(), 1);

  const QVariantMap first_row = backend.Thumbnails().front().toMap();
  const auto        element_id =
      static_cast<sl_element_id_t>(first_row.value("elementId").toUInt());
  const auto image_id = static_cast<image_id_t>(first_row.value("imageId").toUInt());
  ASSERT_NE(element_id, 0);
  ASSERT_NE(image_id, 0);

  ProjectService project(db_path_, meta_path_);
  auto           pipeline_service = std::make_shared<PipelineMgmtService>(project.GetStorageService());
  auto           pipeline_guard   = pipeline_service->LoadPipeline(element_id);
  ASSERT_NE(pipeline_guard, nullptr);
  ASSERT_NE(pipeline_guard->pipeline_, nullptr);

  auto exec = pipeline_guard->pipeline_;
  auto& global_params  = exec->GetGlobalParams();
  auto& loading_stage  = exec->GetStage(PipelineStageName::Image_Loading);
  auto& geometry_stage = exec->GetStage(PipelineStageName::Geometry_Adjustment);

  nlohmann::json raw_params = pipeline_defaults::MakeDefaultRawDecodeParams();
  raw_params["raw"]["gpu_backend"] = "gpu";
  raw_params["raw"]["backend"]     = "alcedo";
  loading_stage.SetOperator(OperatorType::RAW_DECODE, raw_params);

  nlohmann::json crop_params = pipeline_defaults::MakeDefaultCropRotateParams();
  crop_params["crop_rotate"]["enabled"]     = true;
  crop_params["crop_rotate"]["enable_crop"] = true;
  crop_params["crop_rotate"]["angle_degrees"] = 0.0f;
  crop_params["crop_rotate"]["crop_rect"] = {
      {"x", 0.12f},
      {"y", 0.08f},
      {"w", 0.62f},
      {"h", 0.58f},
  };
  geometry_stage.SetOperator(OperatorType::CROP_ROTATE, crop_params, global_params);

  pipeline_guard->dirty_ = true;
  pipeline_service->SavePipeline(pipeline_guard);
  pipeline_service->Sync();

  QSignalSpy thumb_spy(&backend, &AlbumBackend::ThumbnailUpdated);

  backend.SetThumbnailVisible(static_cast<uint>(element_id), static_cast<uint>(image_id), true);

  ASSERT_TRUE(WaitForSignal(thumb_spy, 30000))
      << "Timed out waiting for ThumbnailUpdated after requesting visible thumbnail.";
  ProcessEvents(500);

  const QString first_thumb_url = WaitForThumbnailUrl(backend, element_id, true, 10000);
  ASSERT_FALSE(first_thumb_url.isEmpty())
      << "ThumbnailGridView-style pinning did not produce a thumbUrl.";

  const QVariantMap loaded_row = WaitForThumbnailRow(
      backend, element_id,
      [](const QVariantMap& row) {
        return !row.value("thumbUrl").toString().isEmpty() &&
               !row.value("thumbLoading").toBool() &&
               !row.value("thumbMissingSource").toBool();
      },
      10000);
  ASSERT_FALSE(loaded_row.isEmpty())
      << "Loaded thumbnail row did not clear loading state or unexpectedly marked source missing.";

  backend.SetThumbnailVisible(static_cast<uint>(element_id), static_cast<uint>(image_id), false);
  ProcessEvents(500);

  const QString cleared_thumb_url = WaitForThumbnailUrl(backend, element_id, false, 5000);
  EXPECT_TRUE(cleared_thumb_url.isEmpty())
      << "ThumbnailGridView-style unpinning should clear the visible thumbUrl.";
#endif
}

TEST_F(ThumbnailTests, MissingSourceThumbnailStopsLoadingAndSetsMissingFlag) {
  AlbumBackend backend;
  ASSERT_TRUE(CreateTestProject(backend));

  auto images = CollectRawTestImages("airplane", 1);
  if (images.empty()) {
    images = CollectRawTestImages("still_life", 1);
  }
  ASSERT_FALSE(images.empty()) << "No RAW test image available for missing-source thumbnail test.";

  const auto copied_image = temp_dir_ / images.front().filename();
  std::filesystem::copy_file(images.front(), copied_image,
                             std::filesystem::copy_options::overwrite_existing);

  backend.StartImport(PathsToQStringList({copied_image}));
  WaitForImportFinished(backend);

  ASSERT_FALSE(backend.ImportRunning());
  ASSERT_GE(backend.ShownCount(), 1);

  const QVariantMap first_row = backend.Thumbnails().front().toMap();
  const auto        element_id =
      static_cast<sl_element_id_t>(first_row.value("elementId").toUInt());
  const auto image_id = static_cast<image_id_t>(first_row.value("imageId").toUInt());
  ASSERT_NE(element_id, 0);
  ASSERT_NE(image_id, 0);

  std::filesystem::remove(copied_image);

  QSignalSpy thumb_spy(&backend, &AlbumBackend::ThumbnailUpdated);
  backend.SetThumbnailVisible(static_cast<uint>(element_id), static_cast<uint>(image_id), true);

  ASSERT_TRUE(WaitForSignal(thumb_spy, 10000))
      << "Timed out waiting for ThumbnailUpdated after source file removal.";

  const QVariantMap missing_row = WaitForThumbnailRow(
      backend, element_id,
      [](const QVariantMap& row) {
        return row.value("thumbUrl").toString().isEmpty() &&
               !row.value("thumbLoading").toBool() &&
               row.value("thumbMissingSource").toBool();
      },
      10000);
  ASSERT_FALSE(missing_row.isEmpty())
      << "Missing-source thumbnail row did not settle into the expected error state.";

  backend.SetThumbnailVisible(static_cast<uint>(element_id), static_cast<uint>(image_id), false);
  ProcessEvents(250);
}

}  // namespace alcedo::ui::test
