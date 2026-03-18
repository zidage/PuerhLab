//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "app/sleeve_service.hpp"
#include "type/type.hpp"

namespace puerhlab {

struct AlbumFolderView {
  sl_element_id_t       folder_id_   = 0;
  sl_element_id_t       parent_id_   = 0;
  file_name_t           folder_name_{};
  std::filesystem::path folder_path_{};
  int                   depth_       = 0;
};

struct AlbumFileView {
  sl_element_id_t element_id_       = 0;
  sl_element_id_t parent_folder_id_ = 0;
  image_id_t      image_id_         = 0;
  file_name_t     file_name_{};
};

struct AlbumFolderSnapshot {
  std::vector<AlbumFolderView>                           folders_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t>   parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> path_by_id_{};
};

struct AlbumDeleteResult {
  std::vector<sl_element_id_t> deleted_ids_{};
  std::vector<sl_element_id_t> failed_ids_{};
};

class AlbumBrowseService {
 public:
  explicit AlbumBrowseService(std::shared_ptr<SleeveServiceImpl> sleeve_service)
      : sleeve_service_(std::move(sleeve_service)) {}

  [[nodiscard]] auto ListFolders() const -> AlbumFolderSnapshot;
  [[nodiscard]] auto ListFilesInFolder(sl_element_id_t folder_id) const -> std::vector<AlbumFileView>;
  [[nodiscard]] auto ListFilesInFolders(const std::unordered_set<sl_element_id_t>& folder_ids) const
      -> std::vector<AlbumFileView>;

  [[nodiscard]] auto CreateFolder(sl_element_id_t parent_folder_id, const file_name_t& name)
      -> std::optional<AlbumFolderView>;
  [[nodiscard]] bool DeleteFolder(sl_element_id_t folder_id);
  [[nodiscard]] auto DeleteFiles(const std::vector<sl_element_id_t>& element_ids)
      -> AlbumDeleteResult;

 private:
  std::shared_ptr<SleeveServiceImpl> sleeve_service_{};
};

}  // namespace puerhlab
