//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <QVariantList>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ui/puerhlab_main/album_backend/album_types.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Owns lazy folder-tree state and handles folder CRUD operations.
class FolderController {
 public:
  explicit FolderController(AlbumBackend& backend);

  void               ReloadTree(const std::filesystem::path& preferredFolderPath);
  void               RebuildFolderView();
  void               ApplyFolderSelection(uint folderUiId, bool emitSignal);
  [[nodiscard]] auto CurrentFolderFsPath() const -> std::filesystem::path;
  [[nodiscard]] auto CurrentFolderElementId() const -> std::optional<sl_element_id_t>;
  void               SelectFolder(uint folderUiId);
  void               CreateFolder(const QString& folderName);
  void               DeleteFolder(uint folderUiId);

  [[nodiscard]] auto folders() const -> const QVariantList& { return folders_; }
  [[nodiscard]] auto current_folder_id() const -> uint32_t { return current_folder_ui_id_; }
  [[nodiscard]] auto current_folder_path() const -> const std::filesystem::path& {
    return current_folder_path_;
  }
  [[nodiscard]] auto current_folder_path_text() const -> const QString& {
    return current_folder_path_text_;
  }

  [[nodiscard]] auto folder_entries() const -> const std::vector<ExistingFolderEntry>& {
    return folder_entries_;
  }

  void ClearState();

 private:
  struct FolderNodeState {
    uint32_t              ui_id_         = 0;
    file_name_t           folder_name_{};
    std::filesystem::path folder_path_{};
    int                   depth_         = 0;
    bool                  expanded_      = false;
  };

  [[nodiscard]] auto PathKey(const std::filesystem::path& path) const -> std::wstring;
  void               EnsureRootNode();
  void               ResetTreeState();
  void               EnsurePathExpanded(const std::filesystem::path& folderPath);
  void               LoadChildren(const std::filesystem::path& parentPath);
  void               AppendVisibleEntries(const std::filesystem::path& folderPath,
                                          std::vector<ExistingFolderEntry>& out) const;
  [[nodiscard]] auto TryGetPathForUiId(uint folderUiId) const
      -> std::optional<std::filesystem::path>;
  [[nodiscard]] auto EnsureNode(const std::filesystem::path& folderPath,
                                const file_name_t&           folderName,
                                int                          depth) -> FolderNodeState&;

  AlbumBackend& backend_;

  std::unordered_map<std::wstring, FolderNodeState>          nodes_by_path_{};
  std::unordered_map<std::wstring, std::vector<std::wstring>> child_keys_by_path_{};
  std::unordered_map<uint32_t, std::wstring>                 path_key_by_ui_id_{};
  std::unordered_set<std::wstring>                           loaded_paths_{};
  std::vector<ExistingFolderEntry>                           folder_entries_{};
  QVariantList                                               folders_{};
  std::filesystem::path                                      current_folder_path_{};
  QString                                                    current_folder_path_text_{};
  uint32_t                                                   current_folder_ui_id_     = 0;
  uint32_t                                                   next_folder_ui_id_        = 1;
};

}  // namespace puerhlab::ui
