#include "storage/mapper/sleeve/file_mapper.hpp"

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

namespace puerhlab {
auto FileMapper::FromRawData(std::vector<VarTypes>&& data) -> FileMapperParams {
  if (data.size() != 2) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveFile");
  }

  auto file_id = std::get_if<sl_element_id_t>(&data[0]);
  auto img_id  = std::get_if<image_id_t>(&data[1]);
  if (file_id == nullptr || img_id == nullptr) {
    throw std::runtime_error("Unmatching types occured when parsing the data from the DB");
  }
  return {*file_id, *img_id};
}

void FileMapper::Insert(const FileMapperParams params) {}
};  // namespace puerhlab