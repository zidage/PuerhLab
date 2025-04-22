#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A type of element that contains files or folders of its kind
 *
 */
class SleeveFolder : public SleeveElement {
 private:
  std::unordered_map<file_name_t, sl_element_id_t>                               _contents;
  std::unordered_map<filter_id_t, std::shared_ptr<std::vector<sl_element_id_t>>> _indicies_cache;

  filter_id_t                                                                    _default_filter;

  uint32_t                                                                       _file_count;
  uint32_t                                                                       _folder_count;

 public:
  ElementType _type = ElementType::FOLDER;
  explicit SleeveFolder(sl_element_id_t id, file_name_t element_name);
  ~SleeveFolder();

  auto Copy(sl_element_id_t new_id) -> std::shared_ptr<SleeveElement>;

  void AddElementToMap(const std::shared_ptr<SleeveElement> element);
  void UpdateElementMap(const file_name_t &name, const sl_element_id_t old_id, const sl_element_id_t new_id);
  void CreateIndex(const std::shared_ptr<FilterCombo> filter);
  auto GetElementIdByName(const file_name_t &name) const -> std::optional<sl_element_id_t>;
  auto ListElements() const -> std::shared_ptr<std::vector<sl_element_id_t>>;
  auto Contains(const file_name_t &name) const -> bool;
  void RemoveNameFromMap(const file_name_t &name);

  void IncrementFolderCount();
  void IncrementFileCount();
  void DecrementFolderCount();
  void DecrementFileCount();
  auto ClearFolder() -> bool;
  auto ResetFilters() -> bool;
};
};  // namespace puerhlab