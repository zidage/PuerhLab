#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "image/image_loader.hpp"
#include "sleeve_base.hpp"
#include "sleeve_view.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {
class SleeveManager {
 private:
  std::shared_ptr<SleeveBase>       _base;
  std::shared_ptr<SleeveView>       _view;
  std::shared_ptr<ImagePoolManager> _image_pool;

 public:
  explicit SleeveManager();

  auto GetBase() -> std::shared_ptr<SleeveBase>;
  auto GetView() -> std::shared_ptr<SleeveView>;
  auto LoadToPath(std::vector<image_path_t> img_os_path, sl_path_t dest) -> uint32_t;
};
};  // namespace puerhlab