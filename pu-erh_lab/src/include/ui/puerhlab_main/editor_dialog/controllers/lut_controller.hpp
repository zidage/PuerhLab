#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <QString>

#include "ui/puerhlab_main/editor_dialog/modules/lut_catalog.hpp"

namespace puerhlab::ui::controllers {

struct LutBrowserViewModel {
  QString                             directory_text_{};
  QString                             status_text_{};
  QString                             selected_path_{};
  bool                                can_open_directory_ = false;
  std::vector<lut_catalog::LutCatalogEntry> entries_{};
};

class LutController final {
 public:
  auto Refresh(const std::string& current_lut_path, bool force_refresh = false)
      -> LutBrowserViewModel;
  auto TryResolveSelection(const QString& entry_path) const -> std::optional<std::string>;
  auto directory() const -> const std::filesystem::path& { return catalog_.directory_; }
  auto DefaultLutPath() const -> std::string;

 private:
  auto BuildViewModel(const std::string& current_lut_path) const -> LutBrowserViewModel;

  lut_catalog::LutCatalog catalog_{};
};

}  // namespace puerhlab::ui::controllers
