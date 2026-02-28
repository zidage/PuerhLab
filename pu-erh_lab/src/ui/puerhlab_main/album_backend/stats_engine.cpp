//  Copyright 2026 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "ui/puerhlab_main/album_backend/stats_engine.hpp"

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"

#include <duckdb.h>

#include <format>
#include <string>

namespace puerhlab::ui {

// ── Helper: run a two-column (VARCHAR, BIGINT) GROUP BY query ───────────────

static auto RunGroupByQuery(duckdb_connection conn,
                            const std::string& sql) -> QVariantList {
  QVariantList rows;
  duckdb_result result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    duckdb_destroy_result(&result);
    return rows;
  }

  const auto row_count = duckdb_row_count(&result);
  for (idx_t r = 0; r < row_count; ++r) {
    char* label_raw = duckdb_value_varchar(&result, 0, r);
    const QString label =
        (label_raw && label_raw[0] != '\0')
            ? QString::fromUtf8(label_raw)
            : QStringLiteral("(unknown)");
    if (label_raw) {
      duckdb_free(label_raw);
    }

    const int64_t count = duckdb_value_int64(&result, 1, r);
    rows.push_back(QVariantMap{{"label", label}, {"count", static_cast<int>(count)}});
  }

  duckdb_destroy_result(&result);
  return rows;
}

static auto RunScalarInt64(duckdb_connection conn,
                           const std::string& sql) -> int64_t {
  duckdb_result result;
  if (duckdb_query(conn, sql.c_str(), &result) != DuckDBSuccess) {
    duckdb_destroy_result(&result);
    return 0;
  }
  int64_t value = 0;
  if (duckdb_row_count(&result) > 0) {
    value = duckdb_value_int64(&result, 0, 0);
  }
  duckdb_destroy_result(&result);
  return value;
}

// ── StatsEngine ─────────────────────────────────────────────────────────────

StatsEngine::StatsEngine(AlbumBackend& backend) : backend_(backend) {}

void StatsEngine::RebuildThumbnailView() {
  backend_.thumb_.ReleaseVisibleThumbnailPins();

  QVariantList next;
  next.reserve(static_cast<qsizetype>(backend_.all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : backend_.all_images_) {
    if (!IsImageInCurrentFolder(image)) {
      continue;
    }
    next.push_back(MakeThumbMap(image, index++));
  }

  backend_.visible_thumbnails_ = std::move(next);
  emit backend_.ThumbnailsChanged();
  emit backend_.thumbnailsChanged();
  emit backend_.CountsChanged();
}

void StatsEngine::RefreshStats() {
  auto proj = backend_.project_handler_.project();
  if (!proj) {
    date_stats_.clear();
    camera_stats_.clear();
    lens_stats_.clear();
    total_photo_count_ = 0;
    emit backend_.StatsChanged();
    return;
  }

  try {
    auto guard = proj->GetStorageService()->GetDBController().GetConnectionGuard();
    const auto folder_id = backend_.folder_ctrl_.current_folder_id();

    // ── Base JOIN clause ────────────────────────────────────────────────
    // Mirrors the JOIN used by FilterCombo::GenerateIdSQLOn() so that
    // we query the same set of rows that appear in the thumbnail grid.
    const std::string base_join = std::format(
        "FROM FolderContent fc "
        "JOIN Element e ON fc.element_id = e.id "
        "JOIN FileImage fi ON fi.file_id = e.id "
        "JOIN Image i ON i.id = fi.image_id "
        "WHERE fc.folder_id = {} AND e.type = 0",
        folder_id);

    // ── Total count ─────────────────────────────────────────────────────
    total_photo_count_ = static_cast<int>(RunScalarInt64(
        guard.conn_,
        std::format("SELECT COUNT(*) {}", base_join)));

    // ── By capture date (day) ───────────────────────────────────────────
    date_stats_ = RunGroupByQuery(
        guard.conn_,
        std::format(
            "SELECT CAST(json_extract(i.metadata, '$.DateTimeString') AS DATE)::VARCHAR AS d, "
            "COUNT(*) AS c {} "
            "GROUP BY d ORDER BY d DESC",
            base_join));

    // ── By camera model ─────────────────────────────────────────────────
    camera_stats_ = RunGroupByQuery(
        guard.conn_,
        std::format(
            "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Model'), ''), '(unknown)') AS m, "
            "COUNT(*) AS c {} "
            "GROUP BY m ORDER BY c DESC",
            base_join));

    // ── By lens ─────────────────────────────────────────────────────────
    lens_stats_ = RunGroupByQuery(
        guard.conn_,
        std::format(
            "SELECT COALESCE(NULLIF(json_extract_string(i.metadata, '$.Lens'), ''), '(unknown)') AS l, "
            "COUNT(*) AS c {} "
            "GROUP BY l ORDER BY c DESC",
            base_join));

  } catch (...) {
    // If DB access fails, keep whatever stats we had.
  }

  emit backend_.StatsChanged();
}

bool StatsEngine::IsImageInCurrentFolder(const AlbumItem& image) const {
  return image.parent_folder_id == backend_.folder_ctrl_.current_folder_id();
}

auto StatsEngine::FormatPhotoInfo(int shown, int total) const -> QString {
  if (total <= 0) {
    return "No images loaded.";
  }
  if (shown == total) {
    return QString("Showing %1 images").arg(total);
  }
  return QString("Showing %1 of %2").arg(shown).arg(total);
}

auto StatsEngine::MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap {
  const QString aperture =
      image.aperture > 0.0 ? QString::number(image.aperture, 'f', 1) : "--";
  const QString focal =
      image.focal_length > 0.0 ? QString::number(image.focal_length, 'f', 0) : "--";

  return QVariantMap{
      {"elementId", static_cast<uint>(image.element_id)},
      {"imageId", static_cast<uint>(image.image_id)},
      {"fileName", image.file_name.isEmpty() ? "(unnamed)" : image.file_name},
      {"cameraModel", image.camera_model.isEmpty() ? "Unknown" : image.camera_model},
      {"extension", image.extension.isEmpty() ? "--" : image.extension},
      {"iso", image.iso},
      {"aperture", aperture},
      {"focalLength", focal},
      {"captureDate",
       image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd") : "--"},
      {"rating", image.rating},
      {"tags", image.tags},
      {"accent", image.accent.isEmpty() ? album_util::AccentForIndex(static_cast<size_t>(index))
                                        : image.accent},
      {"thumbUrl", image.thumb_data_url},
  };
}

}  // namespace puerhlab::ui
