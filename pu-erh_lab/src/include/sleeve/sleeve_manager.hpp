#pragma once

#include <memory>

#include "image/image_loader.hpp"
#include "sleeve_base.hpp"
#include "sleeve_view.hpp"
#include "storage/image_pool/image_pool_manager.hpp"

namespace puerhlab {
class SleeveManager {
 private:
  std::shared_ptr<SleeveBase>       _base;
  std::shared_ptr<SleeveView>       _view;
  std::shared_ptr<ImagePoolManager> _buffer_manager;

  std::unique_ptr<ImageLoader>      _image_loader;

 public:
  explicit SleeveManager();

  auto GetBase() -> std::shared_ptr<SleeveBase>;
  auto GetView() -> std::shared_ptr<SleeveView>;
  auto FetchImage() -> std::shared_ptr<Image>;
};
};  // namespace puerhlab