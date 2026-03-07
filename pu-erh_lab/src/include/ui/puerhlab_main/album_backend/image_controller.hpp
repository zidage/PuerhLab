//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QVariantList>
#include <QVariantMap>

#include <vector>

#include "type/type.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Handles single/batch image deletion and related project data cleanup.
class ImageController {
 public:
  explicit ImageController(AlbumBackend& backend);

  auto DeleteImages(const QVariantList& targetEntries) -> QVariantMap;

 private:
  struct DeleteTarget {
    sl_element_id_t element_id_       = 0;
    image_id_t      image_id_         = 0;
    sl_element_id_t parent_folder_id_ = 0;
  };

  [[nodiscard]] auto CollectDeleteTargets(const QVariantList& targetEntries) const
      -> std::vector<DeleteTarget>;
  void RebuildProjectViews(sl_element_id_t preferredFolderId);

  AlbumBackend& backend_;
};

}  // namespace puerhlab::ui
