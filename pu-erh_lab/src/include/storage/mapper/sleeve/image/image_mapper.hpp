#pragma once

#include <cstdint>
#include <memory>

#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
// CREATE TABLE Image (id BIGINT PRIMARY KEY, image_path TEXT, file_name TEXT, type INTEGER,
// metadata JSON);
struct ImageParams {
  image_id_t  id;
  const char* image_path;
  const char* file_name;
  uint32_t    type;
  const char* metadata;
};
class ImageMapper : MapperInterface<ImageMapper, ImageParams, image_id_t>,
                    FieldReflectable<ImageMapper> {
 private:
  static auto                  FromRawData(std::vector<duckorm::VarTypes>&& data) -> ImageParams;
  static constexpr uint32_t    _field_count                                      = 5;
  static constexpr const char* _table_name                                       = "Image";
  static constexpr const char* _prime_key_clause                                 = "id={}";
  static constexpr std::array<duckorm::DuckFieldDesc, _field_count> _field_descs = {
      FIELD(ImageParams, id, UINT32), FIELD(ImageParams, image_path, VARCHAR),
      FIELD(ImageParams, file_name, VARCHAR), FIELD(ImageParams, type, UINT32),
      FIELD(ImageParams, metadata, VARCHAR)};

 public:
  friend struct FieldReflectable<ImageMapper>;
  using MapperInterface::MapperInterface;

  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);
};
}  // namespace puerhlab