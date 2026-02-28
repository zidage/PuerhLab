#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace puerhlab::ui::lens_calib {

struct LensCatalog {
  std::vector<std::string>                        brands_{};
  std::map<std::string, std::vector<std::string>> models_by_brand_{};
};

void SortAndUniqueStrings(std::vector<std::string>* values);
auto ResolveLensCatalogPath() -> std::filesystem::path;
auto LoadLensCatalog() -> LensCatalog;

}  // namespace puerhlab::ui::lens_calib
