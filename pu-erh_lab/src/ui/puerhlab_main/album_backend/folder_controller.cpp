//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/puerhlab_main/album_backend/folder_controller.hpp"

#include <algorithm>

#include "ui/puerhlab_main/album_backend/album_backend.hpp"
#include "ui/puerhlab_main/album_backend/path_utils.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab::ui {

#define PL_TEXT(text, ...)                                                                      \
  i18n::MakeLocalizedText(PUERHLAB_I18N_CONTEXT, QT_TRANSLATE_NOOP(PUERHLAB_I18N_CONTEXT, text) \
                                                     __VA_OPT__(, ) __VA_ARGS__)

FolderController::FolderController(AlbumBackend& backend) : backend_(backend) {
  current_folder_path_      = album_util::RootFsPath();
  current_folder_path_text_ = album_util::RootPathText();
}

auto FolderController::PathKey(const std::filesystem::path& path) const -> std::wstring {
  auto normalized = path.lexically_normal();
  if (normalized.empty()) {
    normalized = album_util::RootFsPath();
  }
  return normalized.generic_wstring();
}

void FolderController::EnsureRootNode() {
  const auto root_path = album_util::RootFsPath();
  const auto key       = PathKey(root_path);
  if (nodes_by_path_.contains(key)) {
    return;
  }

  FolderNodeState root;
  root.ui_id_       = 0;
  root.folder_path_ = root_path;
  root.depth_       = 0;
  root.expanded_    = true;
  nodes_by_path_.emplace(key, root);
  path_key_by_ui_id_[0] = key;
}

void FolderController::ResetTreeState() {
  nodes_by_path_.clear();
  child_keys_by_path_.clear();
  path_key_by_ui_id_.clear();
  loaded_paths_.clear();
  folder_entries_.clear();
  folders_.clear();
  next_folder_ui_id_ = 1;
  EnsureRootNode();
}

auto FolderController::EnsureNode(const std::filesystem::path& folderPath,
                                  const file_name_t& folderName, int depth) -> FolderNodeState& {
  const auto key = PathKey(folderPath);
  if (const auto it = nodes_by_path_.find(key); it != nodes_by_path_.end()) {
    it->second.folder_name_ = folderName;
    it->second.folder_path_ = folderPath;
    it->second.depth_       = depth;
    return it->second;
  }

  FolderNodeState node;
  node.ui_id_                           = next_folder_ui_id_++;
  node.folder_name_                     = folderName;
  node.folder_path_                     = folderPath;
  node.depth_                           = depth;

  auto [it, _]                          = nodes_by_path_.emplace(key, std::move(node));
  path_key_by_ui_id_[it->second.ui_id_] = key;
  return it->second;
}

void FolderController::LoadChildren(const std::filesystem::path& parentPath) {
  std::cout << "[LOG] FolderController: Loading Child for: " << parentPath.string() << std::endl;
  auto proj = backend_.project_handler_.project();
  if (!proj) {
    return;
  }

  auto browse = proj->GetAlbumBrowseService();
  if (!browse) {
    return;
  }

  const auto parent_key = PathKey(parentPath);
  const auto parent_it  = nodes_by_path_.find(parent_key);
  if (parent_it == nodes_by_path_.end()) {
    return;
  }
  if (loaded_paths_.contains(parent_key)) {
    return;
  }

  std::vector<std::wstring> child_keys;
  try {
    const auto folders = browse->ListFolders(parentPath);
    child_keys.reserve(folders.size());
    for (const auto& folder : folders) {
      auto& child =
          EnsureNode(folder.folder_path_, folder.folder_name_, parent_it->second.depth_ + 1);
      child_keys.push_back(PathKey(child.folder_path_));
    }
  } catch (...) {
    return;
  }

  child_keys_by_path_[parent_key] = std::move(child_keys);
  loaded_paths_.insert(parent_key);
}

void FolderController::EnsurePathExpanded(const std::filesystem::path& folderPath) {
  EnsureRootNode();

  auto normalized = folderPath.lexically_normal();
  if (normalized.empty()) {
    normalized = album_util::RootFsPath();
  }

  auto current                             = album_util::RootFsPath();
  auto current_key                         = PathKey(current);
  nodes_by_path_.at(current_key).expanded_ = true;
  LoadChildren(current);

  for (const auto& part : normalized.relative_path()) {
    current /= part;
    const auto next_key = PathKey(current);
    if (!nodes_by_path_.contains(next_key)) {
      LoadChildren(current.parent_path().empty() ? album_util::RootFsPath()
                                                 : current.parent_path());
    }
    auto it = nodes_by_path_.find(next_key);
    if (it == nodes_by_path_.end()) {
      break;
    }
    it->second.expanded_ = true;
    LoadChildren(current);
  }
}

void FolderController::AppendVisibleEntries(const std::filesystem::path&      folderPath,
                                            std::vector<ExistingFolderEntry>& out) const {
  const auto key = PathKey(folderPath);
  const auto it  = nodes_by_path_.find(key);
  if (it == nodes_by_path_.end()) {
    return;
  }

  out.push_back({it->second.ui_id_, it->second.folder_name_, it->second.folder_path_,
                 it->second.depth_, it->second.expanded_});

  if (!it->second.expanded_) {
    return;
  }

  std::cout << "[LOG] FolderController: Finding child for " << conv::ToBytes(key) << std::endl;
  const auto child_it = child_keys_by_path_.find(key);
  if (child_it == child_keys_by_path_.end()) {
    std::cout << "No child for " << conv::ToBytes(key) << std::endl;
    return;
  }

  for (const auto& child_key : child_it->second) {
    std::cout << "[LOG] FolderController: Child for " << conv::ToBytes(key)
              << " found: " << conv::ToBytes(child_key) << std::endl;
    const auto node_it = nodes_by_path_.find(child_key);
    if (node_it == nodes_by_path_.end()) {
      continue;
    }
    AppendVisibleEntries(node_it->second.folder_path_, out);
  }
}

void FolderController::RebuildFolderView() {
  std::vector<ExistingFolderEntry> next_entries;
  AppendVisibleEntries(album_util::RootFsPath(), next_entries);
  folder_entries_ = std::move(next_entries);

  QVariantList next;
  next.reserve(static_cast<qsizetype>(folder_entries_.size()));

  for (const auto& folder : folder_entries_) {
    const QString name = folder.ui_id_ == 0 ? PL_TEXT("Root").Render()
                                            : album_util::WStringToQString(folder.folder_name_);
    next.push_back(QVariantMap{{"folderId", folder.ui_id_},
                               {"name", name},
                               {"depth", folder.depth_},
                               {"path", album_util::FolderPathToDisplay(folder.folder_path_)},
                               {"deletable", folder.ui_id_ != 0},
                               {"expanded", folder.expanded_}});
  }

  folders_ = std::move(next);
  emit backend_.FoldersChanged();
}

auto FolderController::TryGetPathForUiId(uint folderUiId) const
    -> std::optional<std::filesystem::path> {
  const auto key_it = path_key_by_ui_id_.find(folderUiId);
  if (key_it == path_key_by_ui_id_.end()) {
    return std::nullopt;
  }
  const auto node_it = nodes_by_path_.find(key_it->second);
  if (node_it == nodes_by_path_.end()) {
    return std::nullopt;
  }
  return node_it->second.folder_path_;
}

void FolderController::ApplyFolderSelection(uint folderUiId, bool emitSignal) {
  auto path_opt  = TryGetPathForUiId(folderUiId);
  auto next_path = path_opt.has_value() ? path_opt.value() : album_util::RootFsPath();
  EnsurePathExpanded(next_path);

  const auto     key          = PathKey(next_path);
  const auto     it           = nodes_by_path_.find(key);
  const uint32_t next_ui_id   = it != nodes_by_path_.end() ? it->second.ui_id_ : 0;

  const bool     id_changed   = current_folder_ui_id_ != next_ui_id;
  const bool     path_changed = current_folder_path_ != next_path;
  current_folder_ui_id_       = next_ui_id;
  current_folder_path_        = next_path;

  const QString next_path_ui  = album_util::FolderPathToDisplay(current_folder_path_);
  const bool    text_changed  = current_folder_path_text_ != next_path_ui;
  current_folder_path_text_   = next_path_ui;

  std::cout << "[LOG] FolderController: Folder Selected: " << next_path_ui.toStdString()
            << std::endl;

  RebuildFolderView();

  if (emitSignal || id_changed || path_changed || text_changed) {
    emit backend_.FolderSelectionChanged();
    emit backend_.folderSelectionChanged();
  }
}

auto FolderController::CurrentFolderFsPath() const -> std::filesystem::path {
  return current_folder_path_;
}

auto FolderController::CurrentFolderElementId() const -> std::optional<sl_element_id_t> {
  auto proj = backend_.project_handler_.project();
  if (!proj) {
    return std::nullopt;
  }

  auto sleeve = proj->GetSleeveService();
  if (!sleeve) {
    return std::nullopt;
  }

  try {
    const auto folder = sleeve->ResolveFolder(current_folder_path_);
    if (!folder) {
      return std::nullopt;
    }
    return folder->element_id_;
  } catch (...) {
    return std::nullopt;
  }
}

void FolderController::ReloadTree(const std::filesystem::path& preferredFolderPath) {
  ResetTreeState();

  auto target_path = preferredFolderPath.empty() ? album_util::RootFsPath() : preferredFolderPath;
  EnsurePathExpanded(target_path);

  if (!nodes_by_path_.contains(PathKey(target_path))) {
    target_path = album_util::RootFsPath();
    EnsurePathExpanded(target_path);
  }

  auto node_it = nodes_by_path_.find(PathKey(target_path));
  ApplyFolderSelection(node_it != nodes_by_path_.end() ? node_it->second.ui_id_ : 0, true);
}

void FolderController::SelectFolder(uint folderUiId) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading() || !ph.project()) {
    return;
  }

  ApplyFolderSelection(folderUiId, true);
  backend_.stats_.ClearFilters();
  backend_.ReloadCurrentFolder();
  emit backend_.StatsFilterChanged();
}

void FolderController::CreateFolder(const QString& folderName) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState(PL_TEXT("Project is loading. Please wait."), 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState(PL_TEXT("No project is loaded."), 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState(PL_TEXT("Cannot create folder while import is running."), 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState(PL_TEXT("Cannot create folder while export is running."), 0, false);
    return;
  }

  const QString trimmed = folderName.trimmed();
  if (trimmed.isEmpty()) {
    backend_.SetTaskState(PL_TEXT("Folder name cannot be empty."), 0, false);
    return;
  }
  if (trimmed.contains('/') || trimmed.contains('\\')) {
    backend_.SetTaskState(PL_TEXT("Folder name cannot contain '/' or '\\'."), 0, false);
    return;
  }

  auto browse = ph.project()->GetAlbumBrowseService();
  if (!browse) {
    backend_.SetTaskState(PL_TEXT("Folder service is unavailable."), 0, false);
    return;
  }

  const auto created = browse->CreateFolder(current_folder_path_, trimmed.toStdWString());
  if (!created.has_value()) {
    backend_.SetTaskState(PL_TEXT("Failed to create folder."), 0, false);
    return;
  }

  bool save_ok = true;
  try {
    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }
    QString ignored_error;
    if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
      save_ok = false;
    }
  } catch (...) {
    save_ok = false;
  }

  backend_.ReloadFolderTree(current_folder_path_);

  auto msg = PL_TEXT("Created folder %1", album_util::WStringToQString(created->folder_name_));
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
  }
  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, 100, false);
  backend_.ScheduleIdleTaskStateReset(1200);
}

void FolderController::DeleteFolder(uint folderUiId) {
  auto& ph = backend_.project_handler_;
  if (ph.project_loading()) {
    backend_.SetTaskState(PL_TEXT("Project is loading. Please wait."), 0, false);
    return;
  }
  if (!ph.project()) {
    backend_.SetTaskState(PL_TEXT("No project is loaded."), 0, false);
    return;
  }
  auto& ie = backend_.import_export_;
  if (ie.current_import_job() && !ie.current_import_job()->IsCancelationAcked()) {
    backend_.SetTaskState(PL_TEXT("Cannot delete folder while import is running."), 0, false);
    return;
  }
  if (ie.export_inflight()) {
    backend_.SetTaskState(PL_TEXT("Cannot delete folder while export is running."), 0, false);
    return;
  }

  const auto path_opt = TryGetPathForUiId(folderUiId);
  if (!path_opt.has_value() || path_opt.value() == album_util::RootFsPath()) {
    backend_.SetTaskState(PL_TEXT("Root folder cannot be deleted."), 0, false);
    return;
  }

  const auto folder_path = path_opt.value();
  const auto fallback_path =
      folder_path.parent_path().empty() ? album_util::RootFsPath() : folder_path.parent_path();

  auto browse = ph.project()->GetAlbumBrowseService();
  if (!browse || !browse->DeleteFolder(folder_path)) {
    backend_.SetTaskState(PL_TEXT("Failed to delete folder."), 0, false);
    return;
  }

  bool save_ok = true;
  try {
    if (!ph.meta_path().empty()) {
      ph.project()->SaveProject(ph.meta_path());
    }
    QString ignored_error;
    if (!ph.PackageCurrentProjectFiles(&ignored_error)) {
      save_ok = false;
    }
  } catch (...) {
    save_ok = false;
  }

  backend_.ReloadFolderTree(fallback_path);
  backend_.stats_.ClearFilters();
  backend_.ReloadCurrentFolder();
  emit backend_.StatsFilterChanged();

  auto msg = PL_TEXT("Folder deleted.");
  if (!save_ok) {
    msg = PL_TEXT("%1 Project state save failed.", msg.Render());
  }
  backend_.SetServiceMessageForCurrentProject(msg);
  backend_.SetTaskState(msg, 100, false);
  backend_.ScheduleIdleTaskStateReset(1200);
}

void FolderController::ClearState() {
  nodes_by_path_.clear();
  child_keys_by_path_.clear();
  path_key_by_ui_id_.clear();
  loaded_paths_.clear();
  folder_entries_.clear();
  folders_.clear();
  current_folder_path_      = album_util::RootFsPath();
  current_folder_path_text_ = album_util::RootPathText();
  current_folder_ui_id_     = 0;
  next_folder_ui_id_        = 1;
}

}  // namespace puerhlab::ui

#undef PL_TEXT
