//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "app/project_mask_asset_reader.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "storage/store/mask_asset/mask_asset_store.hpp"

namespace alcedo {

ProjectMaskAssetReader::ProjectMaskAssetReader(std::shared_ptr<Storage> storage)
    : storage_(std::move(storage)) {}

auto ProjectMaskAssetReader::Read(const MaskAssetKey& key) const -> MaskAsset {
  if (!storage_) {
    throw std::runtime_error("Mask asset " + std::string{key.Value()} +
                             " cannot be read: the project has no storage");
  }
  auto           db_guard = storage_->GetDatabase().GetConnectionGuard();
  auto           db_lock  = db_guard.Lock();
  MaskAssetStore store(db_guard.conn_);
  return store.Read(key);
}

}  // namespace alcedo
