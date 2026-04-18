//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "ui/alcedo_main/editor_dialog/controllers/lut_controller.hpp"

namespace alcedo::ui::controllers {

auto LutController::Refresh(const std::string& current_lut_path, bool force_refresh)
    -> LutBrowserViewModel {
  catalog_ = lut_catalog::BuildCatalog(current_lut_path, force_refresh);
  return BuildViewModel(current_lut_path);
}

auto LutController::TryResolveSelection(const QString& entry_path) const -> std::optional<std::string> {
  const std::string raw_path = entry_path.toStdString();
  for (const auto& entry : catalog_.entries_) {
    if (entry.path_ != raw_path) {
      continue;
    }
    if (!entry.selectable_) {
      return std::nullopt;
    }
    return entry.path_;
  }
  return std::nullopt;
}

auto LutController::DefaultLutPath() const -> std::string {
  return lut_catalog::DefaultLutPath(catalog_);
}

auto LutController::BuildViewModel(const std::string& current_lut_path) const -> LutBrowserViewModel {
  LutBrowserViewModel model;
  model.directory_text_      = lut_catalog::FormatDirectoryDisplayText(catalog_.directory_);
  model.status_text_         = lut_catalog::CatalogStatusText(catalog_);
  model.can_open_directory_  = catalog_.directory_exists_;
  model.entries_             = catalog_.entries_;

  const int selected_index = lut_catalog::FindEntryIndexForPath(catalog_, current_lut_path);
  if (selected_index >= 0 && selected_index < static_cast<int>(catalog_.entries_.size())) {
    model.selected_path_ = QString::fromStdString(catalog_.entries_[selected_index].path_);
  } else {
    model.selected_path_.clear();
  }
  return model;
}

}  // namespace alcedo::ui::controllers
