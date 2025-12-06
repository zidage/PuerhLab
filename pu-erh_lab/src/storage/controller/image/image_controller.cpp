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
/**
 * @brief Construct a new Image Controller:: Image Controller object
 *
 * @param guard
 */
ImageController::ImageController(ConnectionGuard&& guard)
    : _guard(std::move(guard)), _service(_guard._conn) {}

/**
 * @brief Capture the image pool and insert the parameters into the database.
 *
 * @param image_pool
 */
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

/**
 * @brief Add an image to the database.
 *
 * @param image
 */
void ImageController::AddImage(std::shared_ptr<Image> image) { _service.Insert(image); }

/**
 * @brief Remove an image by its ID.
 *
 * @param remove_id
 */
void ImageController::RemoveImageById(uint32_t remove_id) { _service.RemoveById(remove_id); }

/**
 * @brief Remove an image by its type.
 *
 * @param type
 */
void ImageController::RemoveImageByType(ImageType type) {
  _service.RemoveByClause(std::format("type={}", static_cast<uint32_t>(type)));
}

/**
 * @brief Remove an image by its path.
 *
 * @param path
 */
void ImageController::RemoveImageByPath(const std::wstring& path) {
  _service.RemoveByClause(std::format("image_path={}", conv::ToBytes(path)));
}

/**
 * @brief Get an image by its ID.
 *
 * @param id
 * @return std::shared_ptr<Image>
 */
auto ImageController::GetImageById(image_id_t id) -> std::shared_ptr<Image> {
  auto result = _service.GetImageById(id);
  // Assume the id is unique
  return result[0];
}

/**
 * @brief Get images by their type.
 *
 * @param type
 * @return std::vector<std::shared_ptr<Image>>
 */
auto ImageController::GetImageByType(ImageType type) -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByType(type);
}

/**
 * @brief Get images by their name.
 *
 * @param name
 * @return std::vector<std::shared_ptr<Image>>
 */
auto ImageController::GetImageByName(const std::wstring& name) -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByName(name);
}

/**
 * @brief Get images by their path.
 *
 * @param path
 * @return std::vector<std::shared_ptr<Image>>
 */
auto ImageController::GetImageByPath(const std::filesystem::path path)
    -> std::vector<std::shared_ptr<Image>> {
  return _service.GetImageByPath(path);
}
};  // namespace puerhlab