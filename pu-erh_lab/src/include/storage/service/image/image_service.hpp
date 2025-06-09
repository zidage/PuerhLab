#pragma once

#include <duckdb.h>

#include <codecvt>
#include <filesystem>
#include <memory>
#include <vector>

#include "image/image.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/mapper/image/image_mapper.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImageService {
 private:
  duckdb_connection                                _conn;
  ImageMapper                                      _mapper;

  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;

 public:
  explicit ImageService(duckdb_connection conn) : _conn(conn), _mapper(_conn) {}

  auto ToParams(const Image& source) -> ImageMapperParams;
  auto FromParams(const ImageMapperParams& param) -> std::shared_ptr<Image>;

  void CaptureImagePool(std::shared_ptr<ImagePoolManager> image_pool);
  void InsertImage(const Image& img);

  auto GetImageByPredicate(std::wstring predicate) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageById(image_id_t id) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByName(std::wstring name) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByPath(std::filesystem::path path) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByType(ImageType type) -> std::vector<std::shared_ptr<Image>>;

  void RemoveImageByPredicate(std::wstring predicate);
  void RemoveImageById(image_id_t id);
  void ClearAllImage();

  void UpdateImage(const Image& updated);
};
};  // namespace puerhlab
