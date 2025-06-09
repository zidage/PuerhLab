#include "storage/service/image/image_service.hpp"

#include <memory>

namespace puerhlab {
auto ImageService::ToParams(const Image& source) -> ImageMapperParams {
  return {source._image_id,
          std::make_unique<std::string>(conv.to_bytes(source._image_path.wstring())),
          std::make_unique<std::string>(conv.to_bytes(source._image_name)),
          static_cast<uint32_t>(source._image_type),
          std::make_unique<std::string>(source.ExifToJson())};
}
auto FromParams(const ImageMapperParams& param) -> std::shared_ptr<Image>;
};  // namespace puerhlab