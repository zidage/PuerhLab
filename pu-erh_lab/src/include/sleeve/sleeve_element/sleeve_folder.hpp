#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief A type of element that contains files or folders of its kind
 *
 */
class SleeveFolder : SleeveElement {
  std::unordered_map<sl_element_id_t, std::shared_ptr<SleeveElement>>              _contents;
  std::unordered_map<FilterCombo, std::vector<sl_element_id_t>, FilterComboHasher> _indicies_cache;

  explicit SleeveFolder(sl_element_id_t id);

  void AddElement();
  auto GetElementById(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
  auto GetElementByName(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
  auto ListElements() -> std::vector<std::shared_ptr<SleeveElement>>;
  auto RecursiveListElements() -> std::vector<std::shared_ptr<SleeveElement>>;
  auto RemoveElementById(sl_element_id_t id) -> std::shared_ptr<SleeveElement>;
  auto RemoveElementByName(file_name_t name) -> std::shared_ptr<SleeveElement>;
};
};  // namespace puerhlab