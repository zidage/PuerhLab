#pragma once

#include <QVariantList>

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "ui/puerhlab_main/album_backend/album_types.hpp"

namespace puerhlab::ui {

class AlbumBackend;

/// Owns folder tree state and handles folder CRUD operations.
class FolderController {
 public:
  explicit FolderController(AlbumBackend& backend);

  void RebuildFolderView();
  void ApplyFolderSelection(sl_element_id_t folderId, bool emitSignal);
  [[nodiscard]] auto CurrentFolderFsPath() const -> std::filesystem::path;
  void SelectFolder(uint folderId);
  void CreateFolder(const QString& folderName);
  void DeleteFolder(uint folderId);

  [[nodiscard]] auto folders() const -> const QVariantList& { return folders_; }
  [[nodiscard]] auto current_folder_id() const -> sl_element_id_t { return current_folder_id_; }
  [[nodiscard]] auto current_folder_path_text() const -> const QString& {
    return current_folder_path_text_;
  }

  [[nodiscard]] auto folder_entries() const -> const std::vector<ExistingFolderEntry>& {
    return folder_entries_;
  }
  [[nodiscard]] auto folder_parent_by_id() const
      -> const std::unordered_map<sl_element_id_t, sl_element_id_t>& {
    return folder_parent_by_id_;
  }
  [[nodiscard]] auto folder_path_by_id() const
      -> const std::unordered_map<sl_element_id_t, std::filesystem::path>& {
    return folder_path_by_id_;
  }

  /// Replace folder state wholesale (used after snapshot / project load).
  void SetFolderState(std::vector<ExistingFolderEntry>                          entries,
                      std::unordered_map<sl_element_id_t, sl_element_id_t>     parents,
                      std::unordered_map<sl_element_id_t, std::filesystem::path> paths);
  /// Clear everything.
  void ClearState();

 private:
  AlbumBackend& backend_;

  std::vector<ExistingFolderEntry>                          folder_entries_{};
  std::unordered_map<sl_element_id_t, sl_element_id_t>     folder_parent_by_id_{};
  std::unordered_map<sl_element_id_t, std::filesystem::path> folder_path_by_id_{};
  QVariantList folders_{};
  sl_element_id_t current_folder_id_       = 0;
  QString         current_folder_path_text_ = "\\";
};

}  // namespace puerhlab::ui
