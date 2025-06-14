#include "storage/mapper/sleeve/base/base_mapper.hpp"

#include "type/type.hpp"

namespace puerhlab {
auto BaseMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> BaseMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for Base");
  }
  auto id = std::get_if<sleeve_id_t>(&data[0]);

  if (id == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id};
}
}  // namespace puerhlab