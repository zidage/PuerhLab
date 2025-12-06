#pragma once

#include <duckdb.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "image/image.hpp"
#include "storage/mapper/image/image_mapper.hpp"
#include "storage/service/service_interface.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImageService : public ServiceInterface<ImageService, std::shared_ptr<Image>,
                                             ImageMapperParams, ImageMapper, image_id_t> {
 public:
  using ServiceInterface::ServiceInterface;

  static auto ToParams(const std::shared_ptr<Image> source) -> ImageMapperParams;
  static auto FromParams(ImageMapperParams&& param) -> std::shared_ptr<Image>;

  auto        GetImageById(const image_id_t id) -> std::vector<std::shared_ptr<Image>>;
  auto        GetImageByName(const std::wstring& name) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByPath(const std::filesystem::path path) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByType(const ImageType type) -> std::vector<std::shared_ptr<Image>>;
};
};  // namespace puerhlab
