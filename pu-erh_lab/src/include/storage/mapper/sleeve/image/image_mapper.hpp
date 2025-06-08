#pragma once

#include <cstdint>
#include <memory>

#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "storage/mapper/mapper_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
using namespace duckorm;
struct ImageParams {
  image_id_t  _img_id;
  const char* _img_path;
  const char* _img_name;
  uint32_t    _type;
  const char* _exif_json;
};
class ImageMapper : MapperInterface<ImageParams, image_id_t>, FieldReflectable<ImageMapper> {
 private:
  auto FromRawData(std::vector<VarTypes>&& data) -> ImageParams;

  static constexpr std::array<DuckFieldDesc, 5> kFieldDescs = {
      FIELD(ImageParams, _img_id, UINT32), FIELD(ImageParams, _img_path, VARCHAR),
      FIELD(ImageParams, _img_name, VARCHAR), FIELD(ImageParams, _type, UINT32),
      FIELD(ImageParams, _exif_json, VARCHAR)};

 public:
  using MapperInterface<ImageParams, image_id_t>::MapperInterface;

  friend struct FieldReflectable<ImageMapper>;

  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);

  void Insert(const ImageParams params);
  auto Get(const image_id_t id) -> std::vector<ImageParams>;
  auto Get(const char* where_clause) -> std::vector<ImageParams>;
  void Remove(const image_id_t image_id);
  void Update(const image_id_t id, const ImageParams updated);
};
}  // namespace puerhlab