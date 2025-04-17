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
 protected:
  std::unordered_map<file_name_t, sl_element_id_t>                                 _contents;
  std::unordered_map<FilterCombo, std::vector<sl_element_id_t>, FilterComboHasher> _indicies_cache;

  uint32_t                                                                         _file_count;
  uint32_t                                                                         _folder_count;

 public:
  ElementType _type = ElementType::FOLDER;
  explicit SleeveFolder(sl_element_id_t id, file_name_t element_name);

  void AddElement(std::shared_ptr<SleeveElement>);
  void CreateFilter(FilterCombo &&filter);
  auto GetElementByName(file_name_t name) const -> std::optional<sl_element_id_t>;
  auto Contains(file_name_t name) const -> bool;
  auto ListElements() const -> std::vector<sl_element_id_t>;
  auto RecursiveListElements() const -> std::vector<sl_element_id_t>;
  auto RemoveElementByName(file_name_t name) -> sl_element_id_t;

  void IncrementFolderCount();
  void IncrementFileCount();
  auto ClearFolder() -> bool;
  auto ResetFilters() -> bool;
};
};  // namespace puerhlab