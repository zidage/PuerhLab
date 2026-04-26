//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "app/sleeve_service.hpp"
#include "type/type.hpp"

namespace alcedo {

struct AlbumFolderView {
  file_name_t           folder_name_{};
  std::filesystem::path folder_path_{};
};

struct AlbumFileView {
  sl_element_id_t element_id_       = 0;
  image_id_t      image_id_         = 0;
  file_name_t     file_name_{};
  std::filesystem::path file_path_{};
};

struct AlbumDeleteResult {
  std::vector<AlbumFileView>         deleted_files_{};
  std::vector<std::filesystem::path> failed_paths_{};
  std::vector<sl_element_id_t>       failed_element_ids_{};
};

class AlbumBrowseService {
 public:
  explicit AlbumBrowseService(std::shared_ptr<SleeveServiceImpl> sleeve_service)
      : sleeve_service_(std::move(sleeve_service)) {}

  [[nodiscard]] auto ListFolders(const std::filesystem::path& folder_path) const
      -> std::vector<AlbumFolderView>;
  [[nodiscard]] auto ListFilesInFolder(const std::filesystem::path& folder_path) const
      -> std::vector<AlbumFileView>;

  [[nodiscard]] auto CreateFolder(const std::filesystem::path& parent_folder_path,
                                  const file_name_t&          name)
      -> std::optional<AlbumFolderView>;
  [[nodiscard]] bool DeleteFolder(const std::filesystem::path& folder_path);
  [[nodiscard]] auto DeleteFiles(const std::vector<std::filesystem::path>& file_paths)
      -> AlbumDeleteResult;
  [[nodiscard]] auto DeleteFilesByElementIds(const std::vector<sl_element_id_t>& element_ids)
      -> AlbumDeleteResult;

 private:
  std::shared_ptr<SleeveServiceImpl> sleeve_service_{};
};

}  // namespace alcedo
