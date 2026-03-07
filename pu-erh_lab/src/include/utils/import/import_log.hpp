//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "type/type.hpp"

namespace puerhlab {

struct ImportLogEntry {
  image_id_t      image_id_   = 0;
  sl_element_id_t element_id_ = 0;
  file_name_t     file_name_{};
  bool            metadata_ok_ = false;
};

struct ImportLogSnapshot {
  std::vector<ImportLogEntry> created_{};
  std::vector<image_id_t>     metadata_ok_{};
  std::vector<ImportLogEntry> metadata_failed_{};
};

class ImportLog {
 public:
  void AddPlaceholder(const image_id_t image_id, const sl_element_id_t element_id,
                      const file_name_t& file_name) {
    std::lock_guard<std::mutex> lock(mtx_);
    ImportLogEntry              entry;
    entry.image_id_   = image_id;
    entry.element_id_ = element_id;
    entry.file_name_  = file_name;
    entries_[image_id] = entry;
  }

  void MarkMetadataSuccess(const image_id_t image_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto                        it = entries_.find(image_id);
    if (it == entries_.end()) {
      return;
    }
    it->second.metadata_ok_ = true;
    metadata_ok_.insert(image_id);
  }

  auto Snapshot() const -> ImportLogSnapshot {
    std::lock_guard<std::mutex> lock(mtx_);
    ImportLogSnapshot           snapshot;
    snapshot.created_.reserve(entries_.size());
    snapshot.metadata_ok_.reserve(metadata_ok_.size());
    snapshot.metadata_failed_.reserve(entries_.size());

    for (const auto& [id, entry] : entries_) {
      snapshot.created_.push_back(entry);
      if (entry.metadata_ok_) {
        snapshot.metadata_ok_.push_back(id);
      } else {
        snapshot.metadata_failed_.push_back(entry);
      }
    }
    return snapshot;
  }

 private:
  mutable std::mutex                                      mtx_{};
  std::unordered_map<image_id_t, ImportLogEntry>          entries_{};
  std::unordered_set<image_id_t>                          metadata_ok_{};
};

}  // namespace puerhlab