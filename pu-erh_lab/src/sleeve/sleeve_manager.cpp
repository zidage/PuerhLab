#include "sleeve/sleeve_manager.hpp"

#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_base.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_view.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"
#include "utils/clock/time_provider.hpp"

namespace puerhlab {
/**
 * @brief Ad-hoc constructor for temporary sleeve, e.g. folder preview
 *
 */
SleeveManager::SleeveManager() {
  // Update the application clock
  TimeProvider::Refresh();
  _base       = std::make_shared<SleeveBase>(0);
  _image_pool = std::make_shared<ImagePoolManager>(128, 4);
  _view       = std::make_shared<SleeveView>(_base, _image_pool);
}

/**
 * @brief Return a shared pointer to the sleeve file system interface
 *
 * @return std::shared_ptr<SleeveBase>
 */
auto SleeveManager::GetBase() -> std::shared_ptr<SleeveBase> { return _base; }

/**
 * @brief Return a shared pointer to a sleeve view instance
 *
 * @return std::shared_ptr<SleeveView>
 */
auto SleeveManager::GetView() -> std::shared_ptr<SleeveView> { return _view; }

auto SleeveManager::GetPool() -> std::shared_ptr<ImagePoolManager> { return _image_pool; }

auto SleeveManager::GetImgCount() -> uint32_t { return _image_pool->Capacity(AccessType::META); }

/**
 * @brief Load a batch of images to the destination path
 *
 * @param img_os_paths
 * @param dest
 * @return uint32_t
 */
auto SleeveManager::LoadToPath(std::vector<image_path_t> img_os_paths, sl_path_t dest) -> uint32_t {
  ImageLoader loader{128, 24, 0};
  auto        expected_size = img_os_paths.size();
  auto        total_size    = 0;
  loader.StartLoading(std::move(img_os_paths), DecodeType::SLEEVE_LOADING);
  while (expected_size > 0) {
    std::shared_ptr<Image> loaded  = loader.LoadImage();
    auto                   element = _base->CreateElementToPath(dest, loaded->_image_name, ElementType::FILE);
    if (!element.has_value()) {
      throw std::exception("Error creating element in sleeve");
    }
    std::static_pointer_cast<SleeveFile>(element.value())->SetImage(loaded);
    _image_pool->Insert(loaded);
    total_size++;
    --expected_size;
  }
  return total_size;
}
};  // namespace puerhlab