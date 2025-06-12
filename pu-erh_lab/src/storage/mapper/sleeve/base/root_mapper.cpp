#include "storage/mapper/sleeve/base/base_mapper.hpp"

namespace puerhlab {
auto RootMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> RootMapperParams {
  if (data.size() != _field_count) {
    throw std::runtime_error("Invalid DuckFieldDesc for Base");
  }
  auto id = std::get_if<sl_element_id_t>(&data[0]);

  if (id == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id};
}
}  // namespace puerhlab