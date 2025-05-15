#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve_base.hpp"
#include "type/type.hpp"

namespace puerhlab {

class SleeveView {
  friend class SleeveBase;
  friend class SleeveElement;

 private:
  std::shared_ptr<SleeveBase>               _base;
  std::weak_ptr<SleeveFolder>               _viewing_node;
  sl_path_t                                 _viewing_path;
  std::vector<std::weak_ptr<SleeveElement>> _children;
  std::shared_ptr<ImageLoader>              _loader;

 public:
  SleeveView(std::shared_ptr<SleeveBase> base);
  SleeveView(std::shared_ptr<SleeveBase> base, sl_path_t viewing_path);
  SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<SleeveFolder> viewing_node, sl_path_t viewing_path);

  void UpdateView();
  void LoadPreview(uint32_t range_low, uint32_t range_high);
};
};  // namespace puerhlab