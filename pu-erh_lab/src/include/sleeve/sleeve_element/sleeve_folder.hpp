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
  std::unordered_map<file_name_t, sl_element_id_t>                                 _contents;
  std::unordered_map<FilterCombo, std::vector<sl_element_id_t>, FilterComboHasher> _indicies_cache;

  explicit SleeveFolder(sl_element_id_t id, file_name_t element_name, sl_path_t element_path);

  void AddElement();
  auto GetElementByName(file_name_t name) -> std::shared_ptr<sl_element_id_t>;
  auto ListElements() -> std::vector<std::shared_ptr<sl_element_id_t>>;
  auto RecursiveListElements() -> std::vector<std::shared_ptr<sl_element_id_t>>;
  auto RemoveElementByName(file_name_t name) -> std::shared_ptr<sl_element_id_t>;
};
};  // namespace puerhlab