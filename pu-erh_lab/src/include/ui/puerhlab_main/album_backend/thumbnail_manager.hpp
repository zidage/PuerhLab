//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QString>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

#include "type/type.hpp"


namespace puerhlab::ui {

class AlbumBackend;

/// Manages thumbnail pin reference counts and async data-URL generation.
class ThumbnailManager {
 public:
  explicit ThumbnailManager(AlbumBackend& backend);

  void SetThumbnailVisible(sl_element_id_t elementId, image_id_t imageId, bool visible);
  void RequestThumbnail(sl_element_id_t elementId, image_id_t imageId);
  void UpdateThumbnailState(sl_element_id_t elementId, const QString& dataUrl, bool loading,
                            bool missingSource);
  [[nodiscard]] bool IsThumbnailPinned(sl_element_id_t elementId) const;
 void               RemoveThumbnailState(sl_element_id_t elementId, image_id_t imageId);
  void               ReleaseVisibleThumbnailPins();

 private:
  [[nodiscard]] auto ResolveThumbnailSourcePath(sl_element_id_t elementId,
                                                image_id_t imageId) const
      -> std::filesystem::path;
  [[nodiscard]] static auto PathExists(const std::filesystem::path& path) -> bool;

  AlbumBackend&                                 backend_;
  // TODO: Move pin ref-count tracking into ThumbnailService.
  std::unordered_map<sl_element_id_t, uint32_t> thumbnail_pin_ref_counts_{};
};

}  // namespace puerhlab::ui
