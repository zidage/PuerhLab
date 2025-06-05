#pragma once

#include <duckdb.h>

#include <memory>

#include "image/image.hpp"
#include "mapper_interface.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/duckorm/duckdb_types.hpp"
#include "type/type.hpp"

namespace puerhlab {
using namespace duckorm;
class ImagePoolMapper : MapperInterface<std::shared_ptr<Image>, image_id_t> {
 private:
  auto FromDesc(std::vector<DuckFieldDesc>&& fields) -> std::shared_ptr<Image>;
  auto ToDesc() -> std::vector<DuckFieldDesc>;

 public:
  using MapperInterface::MapperInterface;

  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);

  void Insert(const Image& image);
  auto Get(const image_id_t id) -> std::vector<std::shared_ptr<Image>>;
  auto Get(const char* where_clause) -> std::vector<std::shared_ptr<Image>> = 0;
  void Remove(const image_id_t image_id);
  void Update(const image_id_t id, const Image& updated);
};
}  // namespace puerhlab