#include "ui/puerhlab_main/editor_dialog/controllers/image_controller.hpp"

#include <stdexcept>

#include "image/image.hpp"
#include "io/image/image_loader.hpp"

namespace puerhlab::ui::controllers {

auto LoadImageInputBuffer(const std::shared_ptr<ImagePoolService>& image_pool,
                          image_id_t image_id) -> std::shared_ptr<ImageBuffer> {
  if (!image_pool) {
    throw std::runtime_error("Image controller: image pool service is null");
  }

  auto img_desc = image_pool->Read<std::shared_ptr<Image>>(
      image_id, [](const std::shared_ptr<Image>& img) { return img; });
  auto bytes = ByteBufferLoader::LoadFromImage(img_desc);
  if (!bytes) {
    throw std::runtime_error("Image controller: failed to load image bytes");
  }

  return std::make_shared<ImageBuffer>(std::move(*bytes));
}

}  // namespace puerhlab::ui::controllers
