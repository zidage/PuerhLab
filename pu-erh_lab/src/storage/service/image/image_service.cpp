#include "storage/service/image/image_service.hpp"

#include <utf8.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/image/image_mapper.hpp"
#include "type/type.hpp"
#include "utf8/checked.h"
#include "utils/string/convert.hpp"

namespace puerhlab {

auto ImageService::ToParams(const std::shared_ptr<Image> source) -> ImageMapperParams {
  std::string utf8_path     = conv::ToBytes(source->_image_path.wstring());

  std::string utf8_img_name = conv::ToBytes(source->_image_name);
  return {source->_image_id, std::make_unique<std::string>(utf8_path),
          std::make_unique<std::string>(utf8_img_name), static_cast<uint32_t>(source->_image_type),
          std::make_unique<std::string>(source->ExifToJson())};
}
auto ImageService::FromParams(const ImageMapperParams&& param) -> std::shared_ptr<Image> {
  // TODO: Replace it with ImageFactory once the more fine-grained Image loader is implemented
  auto recovered = std::make_shared<Image>(param.id, std::filesystem::path(*param.image_path),
                                           conv::FromBytes(*param.file_name),
                                           static_cast<ImageType>(param.type));
  recovered->JsonToExif(*param.metadata);
  return recovered;
}

auto ImageService::GetImageById(const image_id_t id) -> std::vector<std::shared_ptr<Image>> {
  std::string predicate = std::format("id={}", id);
  return GetByPredicate(std::move(predicate));
}

auto ImageService::GetImageByName(const std::wstring name) -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate_w = std::format(L"file_name={}", name);
  return GetByPredicate(conv::ToBytes(predicate_w));
}

auto ImageService::GetImageByPath(const std::filesystem::path path)
    -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate_w = std::format(L"image_path={}", path.wstring());

  return GetByPredicate(conv::ToBytes(predicate_w));
}

auto ImageService::GetImageByType(const ImageType type) -> std::vector<std::shared_ptr<Image>> {
  std::string predicate = std::format("type={}", static_cast<uint32_t>(type));
  return GetByPredicate(std::move(predicate));
}

};  // namespace puerhlab