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

  /// Rebuild the thumbnail grid for the current folder (no filtering).
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

 private:
  AlbumBackend& backend_;
  QVariantList  date_stats_{};
  QVariantList  camera_stats_{};
  QVariantList  lens_stats_{};
  int           total_photo_count_ = 0;
};

}  // namespace puerhlab::ui
