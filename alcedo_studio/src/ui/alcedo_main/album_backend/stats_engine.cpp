//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/album_backend/stats_engine.hpp"

#include "ui/alcedo_main/album_backend/album_backend.hpp"
#include "ui/alcedo_main/album_backend/path_utils.hpp"

namespace alcedo::ui {

#define PL_TEXT(text, ...)                                                    \
  i18n::MakeLocalizedText(ALCEDO_I18N_CONTEXT,                              \
                          QT_TRANSLATE_NOOP(ALCEDO_I18N_CONTEXT, text)      \
                              __VA_OPT__(, ) __VA_ARGS__)

namespace {
auto ToStatsRows(const std::vector<alcedo::StatsBucket>& buckets) -> QVariantList {
  QVariantList rows;
  rows.reserve(static_cast<qsizetype>(buckets.size()));
  for (const auto& bucket : buckets) {
    const QString label = bucket.label_.empty()
                              ? PL_TEXT("(unknown)").Render()
                              : QString::fromUtf8(bucket.label_.c_str());
    rows.push_back(QVariantMap{{"label", label}, {"count", bucket.count_}});
  }
  return rows;
}
}  // namespace

StatsEngine::StatsEngine(AlbumBackend& backend) : backend_(backend) {}

void StatsEngine::RebuildThumbnailView() {
  backend_.thumb_.ReleaseVisibleThumbnailPins();

  QVariantList next;
  next.reserve(static_cast<qsizetype>(backend_.view_state_.all_images_.size()));

  int index = 0;
  for (const AlbumItem& image : backend_.view_state_.all_images_) {
    if (!MatchesActiveFilters(image)) {
      continue;
    }
    next.push_back(MakeThumbMap(image, index++));
  }

  backend_.view_state_.visible_thumbnails_ = std::move(next);
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

  auto filter_service = proj->GetSleeveFilterService();
  if (!filter_service) {
    emit backend_.StatsChanged();
    return;
  }

  try {
    const auto folder_id = backend_.folder_ctrl_.CurrentFolderElementId();
    if (!folder_id.has_value()) {
      date_stats_.clear();
      camera_stats_.clear();
      lens_stats_.clear();
      total_photo_count_ = 0;
      emit backend_.StatsChanged();
      return;
    }

    const auto stats = filter_service->BuildFolderStats(folder_id.value());
    total_photo_count_ = stats.total_photo_count_;
    date_stats_        = ToStatsRows(stats.date_stats_);
    camera_stats_      = ToStatsRows(stats.camera_stats_);
    lens_stats_        = ToStatsRows(stats.lens_stats_);
  } catch (...) {
    // Keep previous stats if service query failed.
  }

  emit backend_.StatsChanged();
}

auto StatsEngine::FormatPhotoInfo(int shown, int total) const -> QString {
  if (total <= 0) {
    return PL_TEXT("No images loaded.").Render();
  }
  if (shown == total) {
    return PL_TEXT("Showing %1 images", total).Render();
  }
  return PL_TEXT("Showing %1 of %2", shown, total).Render();
}

auto StatsEngine::MakeThumbMap(const AlbumItem& image, int index) const -> QVariantMap {
  const QString aperture = image.aperture > 0.0 ? QString::number(image.aperture, 'f', 1) : "--";
  const QString focal = image.focal_length > 0.0 ? QString::number(image.focal_length, 'f', 0) : "--";

  return QVariantMap{{"elementId", static_cast<uint>(image.element_id)},
                     {"imageId", static_cast<uint>(image.image_id)},
                     {"fileName", image.file_name.isEmpty() ? PL_TEXT("(unnamed)").Render()
                                                             : image.file_name},
                     {"cameraModel",
                      image.camera_model.isEmpty() ? PL_TEXT("Unknown").Render()
                                                   : image.camera_model},
                     {"extension", image.extension.isEmpty() ? "--" : image.extension},
                     {"iso", image.iso},
                     {"aperture", aperture},
                     {"focalLength", focal},
                     {"captureDate",
                      image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd")
                                                   : "--"},
                     {"rating", image.rating},
                     {"tags", image.tags},
                     {"accent", image.accent.isEmpty()
                                    ? album_util::AccentForIndex(static_cast<size_t>(index))
                                    : image.accent},
                     {"thumbUrl", image.thumb_data_url},
                     {"thumbLoading", image.thumb_loading},
                     {"thumbMissingSource", image.thumb_missing_source}};
}

void StatsEngine::ToggleFilter(const QString& category, const QString& label) {
  if (category == u"date") {
    filter_date_ = (filter_date_ == label) ? QString{} : label;
  } else if (category == u"camera") {
    filter_camera_ = (filter_camera_ == label) ? QString{} : label;
  } else if (category == u"lens") {
    filter_lens_ = (filter_lens_ == label) ? QString{} : label;
  }
}

void StatsEngine::ClearFilters() {
  filter_date_.clear();
  filter_camera_.clear();
  filter_lens_.clear();
}

bool StatsEngine::HasActiveFilter() const {
  return !filter_date_.isEmpty() || !filter_camera_.isEmpty() || !filter_lens_.isEmpty();
}

bool StatsEngine::MatchesActiveFilters(const AlbumItem& image) const {
  if (!HasActiveFilter()) return true;

  if (!filter_date_.isEmpty()) {
    const QString imageDate =
        image.capture_date.isValid() ? image.capture_date.toString("yyyy-MM-dd") : QString{};
    if (filter_date_ == PL_TEXT("(unknown)").Render()) {
      if (image.capture_date.isValid()) return false;
    } else {
      if (imageDate != filter_date_) return false;
    }
  }

  if (!filter_camera_.isEmpty()) {
    if (filter_camera_ == PL_TEXT("(unknown)").Render()) {
      if (!image.camera_model.isEmpty()) return false;
    } else {
      if (image.camera_model != filter_camera_) return false;
    }
  }

  if (!filter_lens_.isEmpty()) {
    if (filter_lens_ == PL_TEXT("(unknown)").Render()) {
      if (!image.lens.isEmpty()) return false;
    } else {
      if (image.lens != filter_lens_) return false;
    }
  }

  return true;
}

}  // namespace alcedo::ui

#undef PL_TEXT
