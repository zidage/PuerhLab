//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <json.hpp>
#include <optional>

#include "ui/alcedo_main/editor_dialog/state.hpp"

namespace alcedo::ui {

enum class PreviewPolicy {
  FastViewport,
  QualityPreview,
  FullFrame,
  GeometryOverlayOnly,
};

enum class CommitPolicy {
  AppendTransaction,
};

struct AdjustmentPreview {
  AdjustmentField field;
  nlohmann::json  params;
  PreviewPolicy   policy = PreviewPolicy::FastViewport;
};

struct AdjustmentCommit {
  AdjustmentField               field;
  std::optional<nlohmann::json> old_params;
  std::optional<nlohmann::json> new_params;
  CommitPolicy                  policy = CommitPolicy::AppendTransaction;
};

}  // namespace alcedo::ui
