//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QString>
#include <QVariantList>
#include <QVariantMap>

#include <filesystem>
#include <vector>

#include "type/type.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Handles single/batch image deletion and related project data cleanup.
class ImageController {
 public:
  struct DeleteTarget {
    sl_element_id_t       element_id_ = 0;
    image_id_t            image_id_   = 0;
    std::filesystem::path file_path_{};
  };

  struct DeleteExecutionResult {
    bool                        success_ = false;
    int                         deleted_count_ = 0;
    int                         failed_count_  = 0;
    std::vector<sl_element_id_t> deleted_element_ids_{};
    std::vector<sl_element_id_t> failed_element_ids_{};
    QString                     message_{};
  };

  explicit ImageController(AlbumBackend& backend);

  auto DeleteImages(const QVariantList& targetEntries) -> QVariantMap;
  auto DeleteTargets(const std::vector<DeleteTarget>& targets) -> DeleteExecutionResult;
  auto GetImageDetails(uint elementId, uint imageId) -> QVariantMap;

 private:
  [[nodiscard]] auto CollectDeleteTargets(const QVariantList& targetEntries) const
      -> std::vector<DeleteTarget>;

  AlbumBackend& backend_;
};

}  // namespace puerhlab::ui
