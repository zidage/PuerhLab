#include "storage/mapper/image/image_mapper.hpp"

#include <cstdint>
#include <variant>

// struct ImageParams {
//   image_id_t  id;
//   const char* image_path;
//   const char* file_name;
//   uint32_t    type;
//   const char* metadata;
// };

namespace puerhlab {
auto ImageMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for Image");
  }
  auto id         = std::get_if<sl_element_id_t>(&data[0]);
  auto image_path = std::get_if<const char*>(&data[1]);
  auto file_name  = std::get_if<const char*>(&data[2]);
  auto type       = std::get_if<uint32_t>(&data[3]);
  auto metadata   = std::get_if<const char*>(&data[4]);

  if (id == nullptr || image_path == nullptr || file_name == nullptr || type == nullptr ||
      metadata == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id, *image_path, *file_name, *type, *metadata};
}
};  // namespace puerhlab