#include <duckdb.h>

#include <memory>

#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImagePoolMapper {
 private:
  duckdb_connection &_con;
  bool               _db_connected = false;

 public:
  ImagePoolMapper(duckdb_connection &con);
  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);

  void AddImage(const Image &image);
  auto GetImage(const image_id_t id) -> std::shared_ptr<Image>;
  void UpdateImage(const Image &image, const image_id_t id);
  void RemoveImage(const image_id_t image_id);
};
}  // namespace puerhlab