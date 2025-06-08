#include "storage/mapper/sleeve/element/element_mapper.hpp"

#include <cstdint>
#include <format>
#include <stdexcept>
#include <variant>

#include "storage/mapper/duckorm/duckdb_orm.hpp"
#include "type/type.hpp"

namespace puerhlab {
// struct ElementMapperParams {
//   sl_element_id_t id;
//   uint32_t        type;
//   const char*     element_name;
//   const char*     added_time;
//   const char*     modified_time;
//   uint32_t        ref_count;
// };
auto ElementMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ElementMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for SleeveElement");
  }
  auto id            = std::get_if<sl_element_id_t>(&data[0]);
  auto type          = std::get_if<uint32_t>(&data[1]);
  auto element_name  = std::get_if<const char*>(&data[2]);
  auto added_time    = std::get_if<const char*>(&data[3]);
  auto modified_time = std::get_if<const char*>(&data[4]);
  auto ref_count     = std::get_if<uint32_t>(&data[5]);

  if (id == nullptr || type == nullptr || element_name == nullptr || added_time == nullptr ||
      modified_time == nullptr || ref_count == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id, *type, *element_name, *added_time, *modified_time, *ref_count};
}
};  // namespace puerhlab