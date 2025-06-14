#include "storage/controller/image/image_controller.hpp"

#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"
#include "storage/service/image/image_service.hpp"
#include "utils/queue/queue.hpp"
#include "utils/string/convert.hpp"

namespace puerhlab {
ImageController::ImageController(ConnectionGuard&& guard) : _guard(guard), _service(_guard._conn) {}

void ImageController::CaptureImagePool(std::shared_ptr<ImagePoolManager> image_pool) {
  ThreadPool                                 thread_pool{8};
  auto&                                      pool = image_pool->GetPool();
  ConcurrentBlockingQueue<ImageMapperParams> converted_params{348};
  for (auto& pool_val : pool) {
    auto img = pool_val.second;
    thread_pool.Submit(
        [img, &converted_params]() { converted_params.push_r(ImageService::ToParams(img)); });
  }

  for (size_t i = 0; i < pool.size(); ++i) {
    auto result = converted_params.pop_r();
    _service.InsertParams(result);
  }
}

void ImageController::AddImage(std::shared_ptr<Image> image) { _service.Insert(image); }

void ImageController::RemoveImageById(uint32_t remove_id) { _service.RemoveById(remove_id); }

void ImageController::RemoveImageByType(ImageType type) {
  _service.RemoveByClause(std::format("type={}", static_cast<uint32_t>(type)));
}

void ImageController::RemoveImageByPath(std::wstring path) {
  _service.RemoveByClause(std::format("image_path={}", conv::ToBytes(path)));
}

auto ImageController::GetImageById(image_id_t id) -> std::shared_ptr<Image> {
  auto result = _service.GetImageById(id);
  // Assume the id is unique
  return result[0];
}

auto ImageController::GetImageByType(ImageType type) -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByType(type);
}

auto ImageController::GetImageByName(std::wstring name) -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByName(name);
}

auto ImageController::GetImageByPath(const std::filesystem::path path)
    -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByPath(path);
}
};  // namespace puerhlab