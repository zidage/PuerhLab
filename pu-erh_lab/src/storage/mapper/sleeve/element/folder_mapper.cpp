#include "storage/mapper/sleeve/element/folder_mapper.hpp"

#include <format>
#include <stdexcept>
#include <variant>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FolderMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> FolderMapperParams {
  if (data.size() != FolderMapper::FieldCount()) {
    throw std::runtime_error("Folder Mapper: Invalid DuckFieldDesc for SleeveFolder");
  }

  auto folder_id  = std::get_if<sl_element_id_t>(&data[0]);
  auto element_id = std::get_if<sl_element_id_t>(&data[1]);

  if (folder_id == nullptr || element_id == nullptr) {
    throw std::runtime_error(
        "Folder Mapper: Unmatching types occured when parsing the data from the DB");
  }

  return {*folder_id, *element_id};
}
};  // namespace puerhlab