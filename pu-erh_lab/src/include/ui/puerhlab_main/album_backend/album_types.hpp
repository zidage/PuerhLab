//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QDate>
#include <QString>
#include <QVariantList>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "type/type.hpp"

namespace puerhlab::ui {

/// Album grid item shown to the user.
struct AlbumItem {
  sl_element_id_t element_id       = 0;
  image_id_t      image_id         = 0;
  std::filesystem::path file_path_{};
  QString         file_name{};
  QString         camera_model{};
  QString         lens{};
  QString         extension{};
  int             iso              = 0;
  double          aperture         = 0.0;
  double          focal_length     = 0.0;
  QDate           capture_date{};
  QDate           import_date{};
  int             rating           = 0;
  QString         tags{};
  QString         accent{};
  QString         thumb_data_url{};
};

/// Per-parameter snapshot used by the embedded editor.
struct EditorState {
  float       exposure_   = 1.0f;
  float       contrast_   = 1.0f;
  float       saturation_ = 0.0f;
  float       tint_       = 0.0f;
  float       blacks_     = 0.0f;
  float       whites_     = 0.0f;
  float       shadows_    = 0.0f;
  float       highlights_ = 0.0f;
  float       sharpen_    = 0.0f;
  float       clarity_    = 0.0f;
  std::string lut_path_{};
};

/// Folder entry in the sleeve tree.
struct ExistingFolderEntry {
  uint32_t              ui_id_       = 0;
  file_name_t           folder_name_{};
  std::filesystem::path folder_path_{};
  int                   depth_       = 0;
  bool                  expanded_    = false;
};

/// UI-only display cache state. No filesystem or DB ownership.
struct AlbumViewState {
  std::vector<AlbumItem> all_images_{};
  QVariantList           visible_thumbnails_{};
};

/// Key for (element, image) pairs used in export.
using ExportTarget = std::pair<sl_element_id_t, image_id_t>;

/// Summary returned after queueing export tasks.
struct ExportQueueBuildResult {
  int     queued_count_  = 0;
  int     skipped_count_ = 0;
  QString first_error_{};
};

}  // namespace puerhlab::ui
