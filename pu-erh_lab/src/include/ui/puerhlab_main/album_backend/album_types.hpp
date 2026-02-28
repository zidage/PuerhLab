#pragma once

#include <QDate>
#include <QString>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/type.hpp"

namespace puerhlab::ui {

/// Album grid item shown to the user.
struct AlbumItem {
  sl_element_id_t element_id       = 0;
  sl_element_id_t parent_folder_id = 0;
  image_id_t      image_id         = 0;
  QString         file_name{};
  QString         camera_model{};
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

/// Result of building a filter tree from the UI model.
struct FilterBuildResult {
  std::optional<FilterNode> node{};
  QString                   error{};
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

/// Sleeve element entry used when loading / snapshotting a project.
struct ExistingAlbumEntry {
  sl_element_id_t element_id_       = 0;
  sl_element_id_t parent_folder_id_ = 0;
  image_id_t      image_id_         = 0;
  file_name_t     file_name_{};
};

/// Folder entry in the sleeve tree.
struct ExistingFolderEntry {
  sl_element_id_t       folder_id_   = 0;
  sl_element_id_t       parent_id_   = 0;
  file_name_t           folder_name_{};
  std::filesystem::path folder_path_{};
  int                   depth_       = 0;
};

/// Full flat snapshot of the sleeve tree at a point in time.
struct ProjectSnapshot {
  std::vector<ExistingAlbumEntry>                          album_entries_{};
  std::vector<ExistingFolderEntry>                         folder_entries_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t>     folder_parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
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
