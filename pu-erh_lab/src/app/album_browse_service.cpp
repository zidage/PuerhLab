//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/album_browse_service.hpp"

#include <algorithm>
#include <unordered_set>

#include "utils/string/convert.hpp"

namespace puerhlab {
namespace {

auto BuildFolderView(const std::filesystem::path&         parent_path,
                     const std::shared_ptr<SleeveFolder>& folder) -> AlbumFolderView {
  AlbumFolderView out;
  out.folder_name_ = folder ? folder->element_name_ : file_name_t{};
  out.folder_path_ = parent_path / out.folder_name_;
  return out;
}

auto BuildFileView(const std::filesystem::path&       parent_path,
                   const std::shared_ptr<SleeveFile>& file) -> AlbumFileView {
  AlbumFileView out;
  out.element_id_ = file ? file->element_id_ : 0;
  out.image_id_   = file ? file->image_id_ : 0;
  out.file_name_  = file ? file->element_name_ : file_name_t{};
  out.file_path_  = parent_path / out.file_name_;
  return out;
}

}  // namespace

auto AlbumBrowseService::ListFolders(const std::filesystem::path& folder_path) const
    -> std::vector<AlbumFolderView> {
  std::vector<AlbumFolderView> folders;
  if (!sleeve_service_) {
    return folders;
  }

  try {
    const auto entries = sleeve_service_->ListFolderEntries(folder_path);
    folders.reserve(entries.size());
    for (const auto& entry : entries) {
      std::cout << "[LOG] AlbumBrowseService: Found file/folder "
                << conv::ToBytes(entry->element_name_) << " in " << folder_path.string()
                << " with size " << entries.size() << std::endl;
      if (!entry || entry->type_ != ElementType::FOLDER || entry->sync_flag_ == SyncFlag::DELETED) {
        continue;
      }
      auto folder = std::dynamic_pointer_cast<SleeveFolder>(entry);
      if (!folder) {
        continue;
      }
      folders.push_back(BuildFolderView(folder_path, folder));
    }
  } catch (...) {
    return {};
  }

  std::sort(folders.begin(), folders.end(),
            [](const AlbumFolderView& lhs, const AlbumFolderView& rhs) {
              if (lhs.folder_name_ != rhs.folder_name_) {
                return lhs.folder_name_ < rhs.folder_name_;
              }
              return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
            });
  return folders;
}

auto AlbumBrowseService::ListFilesInFolder(const std::filesystem::path& folder_path) const
    -> std::vector<AlbumFileView> {
  std::vector<AlbumFileView> files;
  if (!sleeve_service_) {
    return files;
  }

  try {
    const auto entries = sleeve_service_->ListFolderEntries(folder_path);
    files.reserve(entries.size());
    for (const auto& entry : entries) {
      if (!entry || entry->type_ != ElementType::FILE || entry->sync_flag_ == SyncFlag::DELETED) {
        continue;
      }
      auto file = std::dynamic_pointer_cast<SleeveFile>(entry);
      if (!file || file->image_id_ == 0) {
        continue;
      }
      files.push_back(BuildFileView(folder_path, file));
    }
  } catch (...) {
    return {};
  }

  std::sort(files.begin(), files.end(), [](const AlbumFileView& lhs, const AlbumFileView& rhs) {
    if (lhs.file_name_ != rhs.file_name_) {
      return lhs.file_name_ < rhs.file_name_;
    }
    return lhs.element_id_ < rhs.element_id_;
  });
  return files;
}

auto AlbumBrowseService::CreateFolder(const std::filesystem::path& parent_folder_path,
                                      const file_name_t& name) -> std::optional<AlbumFolderView> {
  if (!sleeve_service_) {
    return std::nullopt;
  }

  try {
    const auto result = sleeve_service_->CreateFolder(parent_folder_path, name);
    if (!result.second.success_ || !result.first) {
      return std::nullopt;
    }
    return BuildFolderView(parent_folder_path, result.first);
  } catch (...) {
    return std::nullopt;
  }
}

bool AlbumBrowseService::DeleteFolder(const std::filesystem::path& folder_path) {
  if (!sleeve_service_ || folder_path.empty() || folder_path == std::filesystem::path(L"/")) {
    return false;
  }

  try {
    return sleeve_service_->DeletePath(folder_path).success_;
  } catch (...) {
    return false;
  }
}

auto AlbumBrowseService::DeleteFiles(const std::vector<std::filesystem::path>& file_paths)
    -> AlbumDeleteResult {
  AlbumDeleteResult out;
  if (!sleeve_service_ || file_paths.empty()) {
    return out;
  }

  std::unordered_set<std::wstring> seen;
  seen.reserve(file_paths.size() * 2 + 1);

  for (const auto& path : file_paths) {
    const auto normalized = path.lexically_normal().wstring();
    if (normalized.empty() || !seen.insert(normalized).second) {
      continue;
    }

    try {
      const auto file = sleeve_service_->ResolveFile(path);
      if (!file || file->image_id_ == 0) {
        out.failed_paths_.push_back(path);
        continue;
      }

      const auto view = BuildFileView(path.parent_path(), file);
      const auto sync = sleeve_service_->DeletePath(path);
      if (!sync.success_) {
        out.failed_paths_.push_back(path);
        continue;
      }
      out.deleted_files_.push_back(view);
    } catch (...) {
      out.failed_paths_.push_back(path);
    }
  }

  return out;
}

}  // namespace puerhlab
