#pragma once

#include <memory>

#include "buffer/sleeve_buffer/sleeve_buffer_manager.hpp"
#include "image/image_loader.hpp"
#include "sleeve_base.hpp"
#include "sleeve_view.hpp"

namespace puerhlab {
class SleeveManager {
 private:
  std::shared_ptr<SleeveBase>          _base;
  std::shared_ptr<SleeveView>          _view;
  std::shared_ptr<SleeveBufferManager> _buffer_manager;

  std::unique_ptr<ImageLoader>         _image_loader;

 public:
  explicit SleeveManager();

  auto GetBase() -> std::shared_ptr<SleeveBase>;
  auto GetView() -> std::shared_ptr<SleeveView>;
  auto FetchImage() -> std::shared_ptr<Image>;
};
};  // namespace puerhlab