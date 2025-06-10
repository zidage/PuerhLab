#include "storage/service/image/image_service.hpp"

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

namespace puerhlab {
auto ImageService::ToParams(const Image& source) -> ImageMapperParams {
  return {source._image_id,
          std::make_unique<std::string>(conv.to_bytes(source._image_path.wstring())),
          std::make_unique<std::string>(conv.to_bytes(source._image_name)),
          static_cast<uint32_t>(source._image_type),
          std::make_unique<std::string>(source.ExifToJson())};
}
auto ImageService::FromParams(const ImageMapperParams&& param) -> std::shared_ptr<Image> {
  // TODO: Replace it with ImageFactory once the more fine-grained Image loader is implemented
  auto recovered = std::make_shared<Image>(param.id, std::filesystem::path(*param.image_path),
                                           *param.file_name, static_cast<ImageType>(param.type));
  recovered->JsonToExif(*param.metadata);
  return recovered;
}

void ImageService::InsertImage(const Image& img) {
  auto param = ToParams(img);
  _mapper.Insert(std::move(param));
}

void ImageService::InsertImageParams(const ImageMapperParams& param) {
  _mapper.Insert(std::move(param));
}

auto ImageService::GetImageByPredicate(const std::wstring predicate)
    -> std::vector<std::shared_ptr<Image>> {
  auto                                param_results = _mapper.Get(conv.to_bytes(predicate).c_str());
  std::vector<std::shared_ptr<Image>> image_results;
  image_results.resize(param_results.size());
  size_t idx = 0;
  for (auto& param : param_results) {
    image_results[idx] = FromParams(std::move(param));
    ++idx;
  }
  return image_results;
}

auto ImageService::GetImageById(const image_id_t id) -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate = std::format(L"id={}", id);
  return GetImageByPredicate(predicate);
}

auto ImageService::GetImageByName(const std::wstring name) -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate = std::format(L"file_name={}", name);
  return GetImageByPredicate(predicate);
}

auto ImageService::GetImageByPath(const std::filesystem::path path)
    -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate = std::format(L"image_path={}", path.wstring());
  return GetImageByPredicate(predicate);
}

auto ImageService::GetImageByType(const ImageType type) -> std::vector<std::shared_ptr<Image>> {
  std::wstring predicate = std::format(L"type={}", static_cast<uint32_t>(type));
  return GetImageByPredicate(predicate);
}

void ImageService::RemoveImageById(const image_id_t id) { _mapper.Remove(id); }

void ImageService::UpdateImage(const Image& updated) {
  _mapper.Update(updated._image_id, ToParams(updated));
}

void ImageService::CaptureImagePool(std::shared_ptr<ImagePoolManager> image_pool) {
  ThreadPool                           thread_pool{8};
  auto&                                pool = image_pool->GetPool();
  BlockingMPMCQueue<ImageMapperParams> converted_params{348};
  for (auto& pool_val : pool) {
    auto img = pool_val.second;
    thread_pool.Submit([img, &converted_params, this]() { converted_params.push(ToParams(*img)); });
  }

  for (size_t i = 0; i < pool.size(); ++i) {
    auto result = converted_params.pop();
    InsertImageParams(result);
  }
}
};  // namespace puerhlab