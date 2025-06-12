#include "storage/mapper/image/image_mapper.hpp"

#include <cstdint>
#include <memory>
#include <variant>

// struct ImageParams {
//   image_id_t  id;
//   const char* image_path;
//   const char* file_name;
//   uint32_t    type;
//   const char* metadata;
// };

namespace puerhlab {
auto ImageMapper::FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageMapperParams {
  if (data.size() != FieldCount()) {
    throw std::runtime_error("Invalid DuckFieldDesc for Image");
  }
  auto id         = std::get_if<sl_element_id_t>(&data[0]);
  auto image_path = std::get_if<std::unique_ptr<std::string>>(&data[1]);
  auto file_name  = std::get_if<std::unique_ptr<std::string>>(&data[2]);
  auto type       = std::get_if<uint32_t>(&data[3]);
  auto metadata   = std::get_if<std::unique_ptr<std::string>>(&data[4]);

  if (id == nullptr || image_path == nullptr || file_name == nullptr || type == nullptr ||
      metadata == nullptr) {
    throw std::runtime_error("Encounting unmatching types when parsing the data from the DB");
  }
  return {*id, std::move(*image_path), std::move(*file_name), *type, std::move(*metadata)};
}
};  // namespace puerhlab