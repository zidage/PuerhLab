#include "sleeve/sleeve_element/sleeve_folder.hpp"

#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_element_factory.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Sleeve Folder:: Sleeve Folder object
 *
 * @param id
 * @param element_name
 */
SleeveFolder::SleeveFolder(sl_element_id_t id, file_name_t element_name)
    : SleeveElement(id, element_name), _file_count(0), _folder_count(0) {
  _indicies_cache[_default_filter] = {};
  _type                            = ElementType::FOLDER;
}

SleeveFolder::~SleeveFolder() {}

auto SleeveFolder::Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement> {
  auto copy       = std::make_shared<SleeveFolder>(new_id, _element_name);
  // Copy the name-id map and filter map
  copy->_contents = {_contents};
  // Only copy indicies cache under default filter
  copy->_indicies_cache[copy->_default_filter] =
      std::vector<sl_element_id_t>(_indicies_cache.at(_default_filter));
  return copy;
}
/**
 * @brief Add an element reference to the folder
 *
 * @param element
 */
void SleeveFolder::AddElementToMap(const std::shared_ptr<SleeveElement> element) {
  _contents[element->_element_name] = element->_element_id;
  _indicies_cache[_default_filter].push_back(element->_element_id);
  // Once a pinned element is added to the current folder, current folder also becomes pinned
  _pinned |= element->_pinned;
  element->IncrementRefCount();
}

/**
 * @brief Update a name-id mapping
 *
 * @param name
 * @param old_id
 * @param new_id
 */
void SleeveFolder::UpdateElementMap(const file_name_t& name, const sl_element_id_t old_id,
                                    const sl_element_id_t new_id) {
  _contents.erase(name);
  _contents[name]  = new_id;
  auto default_filter = _indicies_cache[_default_filter];

  for (auto& id : default_filter) {
    if (id == old_id) {
      id = new_id;
      break;
    }
  }
}


/**
 * @brief Get an element's id from the _contents table
 *
 * @param name
 * @return std::optional<sl_element_id_t>
 */
auto SleeveFolder::GetElementIdByName(const file_name_t& name) const
    -> std::optional<sl_element_id_t> {
  if (!Contains(name)) {
    return std::nullopt;
  }
  return _contents.at(name);
}

/**
 * @brief List all the elements within this folder
 *
 * @return std::shared_ptr<std::vector<sl_element_id_t>>
 */
auto SleeveFolder::ListElements() const -> const std::vector<sl_element_id_t>& {
  const auto& default_list = _indicies_cache.at(_default_filter);
  return default_list;
}


auto SleeveFolder::Clear() -> bool {
  // TODO: Add Implementation
  _contents.clear();
  _indicies_cache.clear();

  return true;
}

/**
 * @brief Check whether the folder contains the element of the given name
 *
 * @param name
 * @return true
 * @return false
 */
auto SleeveFolder::Contains(const file_name_t& name) const -> bool {
  return _contents.count(name) != 0;
}

/**
 * @brief Remove a name-id mapping from the _contents table
 *
 * @param name
 * @return sl_element_id_t
 */
void SleeveFolder::RemoveNameFromMap(const file_name_t& name) {
  auto removed_id = _contents.at(name);
  // Also remove from default filter index
  auto& default_index = _indicies_cache[_default_filter];
  default_index.erase(std::remove(default_index.begin(), default_index.end(), removed_id),
                      default_index.end());
  _contents.erase(name);
}

void SleeveFolder::CreateIndex(const std::vector<std::shared_ptr<SleeveElement>>& filtered_elements,
                               const filter_id_t filter_id) {
  std::vector<sl_element_id_t> new_index;
  for (const auto& element : filtered_elements) {
    new_index.push_back(element->_element_id);
  }
  _indicies_cache[filter_id] = new_index;
}

auto SleeveFolder::ListElementsByFilter(const filter_id_t filter_id) const -> const std::vector<sl_element_id_t>& {
  if (!HasFilterIndex(filter_id)) {
    throw std::runtime_error("Filter index not found in folder.");
  }
  return _indicies_cache.at(filter_id);
}


void SleeveFolder::IncrementFileCount() { ++_file_count; }

void SleeveFolder::DecrementFileCount() { --_file_count; }

void SleeveFolder::IncrementFolderCount() { ++_folder_count; }

void SleeveFolder::DecrementFolderCount() { --_folder_count; }

auto SleeveFolder::ContentSize() -> size_t { return _contents.size(); }
};  // namespace puerhlab