#pragma once

#include <memory>

#include "app/image_pool_service.hpp"
#include "image/image_buffer.hpp"

namespace puerhlab::ui::controllers {

auto LoadImageInputBuffer(const std::shared_ptr<ImagePoolService>& image_pool,
                          image_id_t image_id) -> std::shared_ptr<ImageBuffer>;

}  // namespace puerhlab::ui::controllers
