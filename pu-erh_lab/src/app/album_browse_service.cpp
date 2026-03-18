//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/album_browse_service.hpp"

#include <algorithm>
#include <stdexcept>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_filesystem.hpp"

namespace puerhlab {
namespace {

auto RootFsPath() -> std::filesystem::path { return std::filesystem::path(L"/"); }

struct FolderVisit {
  sl_element_id_t       folder_id_  = 0;
  sl_element_id_t       parent_id_  = 0;
  std::filesystem::path folder_path_{};
  int                   depth_      = 0;
};

auto BuildFolderSnapshot(FileSystem& fs) -> AlbumFolderSnapshot {
  AlbumFolderSnapshot snapshot;

  std::shared_ptr<SleeveElement> root_element;
  const auto                     root_path = RootFsPath();
  try {
    root_element = fs.Get(root_path, false);
  } catch (...) {
    root_element.reset();
  }

  if (!root_element || root_element->type_ != ElementType::FOLDER ||
      root_element->sync_flag_ == SyncFlag::DELETED) {
    return snapshot;
  }

  const auto root_id = root_element->element_id_;
  std::vector<FolderVisit> stack{{root_id, root_id, root_path, 0}};
  std::unordered_set<sl_element_id_t> visited;
  visited.reserve(4096);

  while (!stack.empty()) {
    const auto visit = stack.back();
    stack.pop_back();

    if (!visited.insert(visit.folder_id_).second) {
      continue;
    }

    std::shared_ptr<SleeveElement> folder_element;
    try {
      folder_element = fs.Get(visit.folder_path_, false);
    } catch (...) {
      try {
        folder_element = fs.Get(visit.folder_id_);
      } catch (...) {
        continue;
      }
    }

    if (!folder_element || folder_element->sync_flag_ == SyncFlag::DELETED ||
        folder_element->type_ != ElementType::FOLDER) {
      continue;
    }

    AlbumFolderView folder;
    folder.folder_id_ = visit.folder_id_;
    if (visit.folder_id_ == root_id) {
      folder.parent_id_   = root_id;
      folder.folder_name_ = L"";
      folder.folder_path_ = root_path;
      folder.depth_       = 0;
    } else {
      folder.parent_id_   = visit.parent_id_;
      folder.folder_name_ = folder_element->element_name_;
      folder.folder_path_ = visit.folder_path_;
      folder.depth_       = visit.depth_;
    }

    snapshot.folders_.push_back(folder);
    snapshot.parent_by_id_[folder.folder_id_] = folder.parent_id_;
    snapshot.path_by_id_[folder.folder_id_]   = folder.folder_path_;

    std::vector<sl_element_id_t> children;
    try {
      children = fs.ListFolderContent(visit.folder_id_);
    } catch (...) {
      continue;
    }

    std::vector<std::shared_ptr<SleeveElement>> child_elements;
    child_elements.reserve(children.size());
    for (const auto child_id : children) {
      if (child_id == visit.folder_id_) {
        continue;
      }
      try {
        auto child = fs.Get(child_id);
        if (!child || child->sync_flag_ == SyncFlag::DELETED) {
          continue;
        }
        child_elements.push_back(std::move(child));
      } catch (...) {
      }
    }

    std::sort(child_elements.begin(), child_elements.end(),
              [](const std::shared_ptr<SleeveElement>& lhs,
                 const std::shared_ptr<SleeveElement>& rhs) {
                if (lhs->type_ != rhs->type_) {
                  return lhs->type_ == ElementType::FOLDER;
                }
                return lhs->element_name_ < rhs->element_name_;
              });

    for (auto it = child_elements.rbegin(); it != child_elements.rend(); ++it) {
      const auto& child = *it;
      if (child->type_ != ElementType::FOLDER) {
        continue;
      }
      stack.push_back({child->element_id_, visit.folder_id_,
                       visit.folder_path_ / child->element_name_, visit.depth_ + 1});
    }
  }

  std::sort(snapshot.folders_.begin(), snapshot.folders_.end(),
            [](const AlbumFolderView& lhs, const AlbumFolderView& rhs) {
              if (lhs.folder_id_ == 0 || rhs.folder_id_ == 0) {
                return lhs.folder_id_ == 0;
              }
              if (lhs.folder_path_ != rhs.folder_path_) {
                return lhs.folder_path_.generic_wstring() < rhs.folder_path_.generic_wstring();
              }
              return lhs.folder_id_ < rhs.folder_id_;
            });

  return snapshot;
}

auto ListFolderFiles(FileSystem& fs, sl_element_id_t folder_id) -> std::vector<AlbumFileView> {
  std::vector<AlbumFileView> files;

  std::vector<sl_element_id_t> children;
  try {
    children = fs.ListFolderContent(folder_id);
  } catch (...) {
    return files;
  }

  files.reserve(children.size());
  for (const auto child_id : children) {
    try {
      const auto child = fs.Get(child_id);
      if (!child || child->sync_flag_ == SyncFlag::DELETED || child->type_ != ElementType::FILE) {
        continue;
      }
      const auto file = std::dynamic_pointer_cast<SleeveFile>(child);
      if (!file || file->image_id_ == 0) {
        continue;
      }
      files.push_back({file->element_id_, folder_id, file->image_id_, file->element_name_});
    } catch (...) {
    }
  }

  std::sort(files.begin(), files.end(), [](const AlbumFileView& lhs, const AlbumFileView& rhs) {
    if (lhs.file_name_ != rhs.file_name_) {
      return lhs.file_name_ < rhs.file_name_;
    }
    return lhs.element_id_ < rhs.element_id_;
  });

  return files;
}

}  // namespace

auto AlbumBrowseService::ListFolders() const -> AlbumFolderSnapshot {
  if (!sleeve_service_) {
    return {};
  }

  try {
    return sleeve_service_->Read<AlbumFolderSnapshot>(
        [](FileSystem& fs) { return BuildFolderSnapshot(fs); });
  } catch (...) {
    return {};
  }
}

auto AlbumBrowseService::ListFilesInFolder(sl_element_id_t folder_id) const
    -> std::vector<AlbumFileView> {
  if (!sleeve_service_) {
    return {};
  }

  try {
    return sleeve_service_->Read<std::vector<AlbumFileView>>(
        [folder_id](FileSystem& fs) { return ListFolderFiles(fs, folder_id); });
  } catch (...) {
    return {};
  }
}

auto AlbumBrowseService::ListFilesInFolders(const std::unordered_set<sl_element_id_t>& folder_ids) const
    -> std::vector<AlbumFileView> {
  if (!sleeve_service_ || folder_ids.empty()) {
    return {};
  }

  try {
    return sleeve_service_->Read<std::vector<AlbumFileView>>([&folder_ids](FileSystem& fs) {
      std::vector<AlbumFileView> files;
      for (const auto folder_id : folder_ids) {
        auto one = ListFolderFiles(fs, folder_id);
        files.insert(files.end(), one.begin(), one.end());
      }
      std::sort(files.begin(), files.end(), [](const AlbumFileView& lhs, const AlbumFileView& rhs) {
        if (lhs.parent_folder_id_ != rhs.parent_folder_id_) {
          return lhs.parent_folder_id_ < rhs.parent_folder_id_;
        }
        if (lhs.file_name_ != rhs.file_name_) {
          return lhs.file_name_ < rhs.file_name_;
        }
        return lhs.element_id_ < rhs.element_id_;
      });
      return files;
    });
  } catch (...) {
    return {};
  }
}

auto AlbumBrowseService::CreateFolder(sl_element_id_t parent_folder_id, const file_name_t& name)
    -> std::optional<AlbumFolderView> {
  if (!sleeve_service_) {
    return std::nullopt;
  }

  try {
    const auto result = sleeve_service_->Write<AlbumFolderView>(
        [parent_folder_id, name](FileSystem& fs) -> AlbumFolderView {
          const auto snapshot = BuildFolderSnapshot(fs);
          std::filesystem::path parent_path = RootFsPath();
          sl_element_id_t parent_id = parent_folder_id;
          int parent_depth = 0;
          if (parent_folder_id != 0) {
            const auto it = snapshot.path_by_id_.find(parent_folder_id);
            if (it == snapshot.path_by_id_.end()) {
              throw std::runtime_error("Parent folder does not exist.");
            }
            parent_path = it->second;
            for (const auto& folder : snapshot.folders_) {
              if (folder.folder_id_ == parent_folder_id) {
                parent_depth = folder.depth_;
                break;
              }
            }
          }

          const auto created = fs.Create(parent_path, name, ElementType::FOLDER);
          if (!created || created->type_ != ElementType::FOLDER) {
            throw std::runtime_error("Failed to create folder.");
          }

          AlbumFolderView out;
          out.folder_id_   = created->element_id_;
          out.parent_id_   = parent_id;
          out.folder_name_ = created->element_name_;
          out.folder_path_ = parent_path / created->element_name_;
          out.depth_       = parent_depth + 1;
          return out;
        });

    if (!result.second.success_) {
      return std::nullopt;
    }
    return result.first;
  } catch (...) {
    return std::nullopt;
  }
}

bool AlbumBrowseService::DeleteFolder(sl_element_id_t folder_id) {
  if (!sleeve_service_ || folder_id == 0) {
    return false;
  }

  try {
    const auto result = sleeve_service_->Write<void>([folder_id](FileSystem& fs) {
      const auto snapshot = BuildFolderSnapshot(fs);
      const auto it = snapshot.path_by_id_.find(folder_id);
      if (it == snapshot.path_by_id_.end()) {
        throw std::runtime_error("Folder does not exist.");
      }
      fs.Delete(it->second);
    });
    return result.success_;
  } catch (...) {
    return false;
  }
}

auto AlbumBrowseService::DeleteFiles(const std::vector<sl_element_id_t>& element_ids)
    -> AlbumDeleteResult {
  AlbumDeleteResult out;
  if (!sleeve_service_ || element_ids.empty()) {
    return out;
  }

  try {
    const auto result = sleeve_service_->Write<AlbumDeleteResult>(
        [element_ids](FileSystem& fs) {
          AlbumDeleteResult deleted;
          const auto snapshot = BuildFolderSnapshot(fs);

          std::unordered_map<sl_element_id_t, std::filesystem::path> parent_path_by_file_id;
          parent_path_by_file_id.reserve(2048);

          for (const auto& folder : snapshot.folders_) {
            std::vector<sl_element_id_t> children;
            try {
              children = fs.ListFolderContent(folder.folder_id_);
            } catch (...) {
              continue;
            }
            for (const auto child_id : children) {
              try {
                const auto child = fs.Get(child_id);
                if (!child || child->sync_flag_ == SyncFlag::DELETED ||
                    child->type_ != ElementType::FILE) {
                  continue;
                }
                parent_path_by_file_id[child->element_id_] = folder.folder_path_;
              } catch (...) {
              }
            }
          }

          std::unordered_set<sl_element_id_t> seen;
          seen.reserve(element_ids.size() * 2 + 1);

          for (const auto element_id : element_ids) {
            if (element_id == 0 || !seen.insert(element_id).second) {
              continue;
            }

            try {
              const auto parent_it = parent_path_by_file_id.find(element_id);
              if (parent_it == parent_path_by_file_id.end()) {
                deleted.failed_ids_.push_back(element_id);
                continue;
              }

              const auto element = fs.Get(element_id);
              if (!element || element->type_ != ElementType::FILE) {
                deleted.failed_ids_.push_back(element_id);
                continue;
              }

              fs.Delete(parent_it->second / element->element_name_);
              deleted.deleted_ids_.push_back(element_id);
            } catch (...) {
              deleted.failed_ids_.push_back(element_id);
            }
          }

          return deleted;
        });

    if (!result.second.success_) {
      out.failed_ids_ = element_ids;
      return out;
    }
    return result.first;
  } catch (...) {
    out.failed_ids_ = element_ids;
    return out;
  }
}

}  // namespace puerhlab
