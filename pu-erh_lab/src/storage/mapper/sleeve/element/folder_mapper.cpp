#include "storage/mapper/sleeve/element/folder_mapper.hpp"

#include <format>
#include <stdexcept>
#include <variant>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FolderMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> FolderMapperParams {
  if (data.size() != 2) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFolder");
  }

  auto folder_id    = std::get_if<sl_element_id_t>(&data[0]);
  auto element_name = std::get_if<const char*>(&data[1]);
  auto element_id   = std::get_if<sl_element_id_t>(&data[2]);

  if (folder_id == nullptr || element_name == nullptr || element_id == nullptr) {
    throw std::runtime_error("Unmatching types occured when parsing the data from the DB");
  }

  return {*folder_id, *element_name, *element_id};
}
};  // namespace puerhlab