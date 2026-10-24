//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "app/editor_session_ports.hpp"
#include "app/pipeline_service.hpp"
#include "edit/mask/mask_asset.hpp"

namespace alcedo::ui {

/**
 * @brief AI Mask asset store of the editor over the project PipelineMgmtService (SM5 M18).
 *
 * Owns no state: each call resolves the service of the open project and forwards to it, which
 * takes the database connection at the call. Exceptions become the error text.
 */
class EditorSessionMaskAssetPort final : public alcedo::IEditorMaskAssetPort {
 public:
  /// @p pipeline_service resolves the service of the open project; null when no project is open.
  explicit EditorSessionMaskAssetPort(
      std::function<std::shared_ptr<alcedo::PipelineMgmtService>()> pipeline_service);

  auto PutMaskAsset(const alcedo::MaskAsset& asset, std::string* error) -> bool override;
  auto ReadMaskAsset(const alcedo::MaskAssetKey& key, alcedo::MaskAsset* asset, std::string* error)
      -> bool override;

 private:
  std::function<std::shared_ptr<alcedo::PipelineMgmtService>()> pipeline_service_;
};

}  // namespace alcedo::ui
