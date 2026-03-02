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

#pragma once

#include <QString>
#include <QVariantList>
#include <QVariantMap>

#include "ui/puerhlab_main/album_backend/album_types.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Runs SQL aggregate queries against DuckDB after import / folder change,
/// and manages the thumbnail view rebuild.
class StatsEngine {
 public:
  explicit StatsEngine(AlbumBackend& backend);

  /// Rebuild the thumbnail grid for the current folder, applying active stats filters.
  void RebuildThumbnailView();

  /// Execute GROUP BY aggregate queries and update stats properties.
  void RefreshStats();

  [[nodiscard]] bool IsImageInCurrentFolder(const AlbumItem& image) const;
  [[nodiscard]] auto FormatPhotoInfo(int shown, int total) const -> QString;
  [[nodiscard]] auto MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap;

  [[nodiscard]] auto date_stats() const -> const QVariantList& { return date_stats_; }
  [[nodiscard]] auto camera_stats() const -> const QVariantList& { return camera_stats_; }
  [[nodiscard]] auto lens_stats() const -> const QVariantList& { return lens_stats_; }
  [[nodiscard]] int  total_photo_count() const { return total_photo_count_; }

  // ── Stats-bar filter ────────────────────────────────────────────────
  /// Toggle a single-category filter. category is "date", "camera", or "lens".
  /// If the current filter value for that category equals label, it is cleared
  /// (toggle off); otherwise it is set to label.
  void ToggleFilter(const QString& category, const QString& label);

  /// Clear all active stats-bar filters.
  void ClearFilters();

  [[nodiscard]] bool HasActiveFilter() const;
  [[nodiscard]] const QString& filter_date() const { return filter_date_; }
  [[nodiscard]] const QString& filter_camera() const { return filter_camera_; }
  [[nodiscard]] const QString& filter_lens() const { return filter_lens_; }

 private:
  /// Returns true if the image passes all currently active stats-bar filters.
  [[nodiscard]] bool MatchesActiveFilters(const AlbumItem& image) const;
  AlbumBackend& backend_;
  QVariantList  date_stats_{};
  QVariantList  camera_stats_{};
  QVariantList  lens_stats_{};
  int           total_photo_count_ = 0;

  // Active stats-bar filter values (empty = no filter for that category)
  QString       filter_date_{};
  QString       filter_camera_{};
  QString       filter_lens_{};
};

}  // namespace puerhlab::ui
