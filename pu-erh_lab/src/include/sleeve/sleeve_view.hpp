#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve_base.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {

struct DisplayingImage {
 private:
  std::shared_ptr<Image> _displaying;
  bool                   _require_thumb;
  bool                   _require_full;

 public:
  DisplayingImage(std::weak_ptr<Image> _displaying, bool _require_thumb, bool _require_full);
  DisplayingImage(std::shared_ptr<Image> _displaying, bool _require_thumb, bool _require_full);
  ~DisplayingImage();
};

class SleeveView {
  friend class SleeveBase;
  friend class SleeveElement;

 private:
  std::shared_ptr<SleeveBase>               _base;
  std::weak_ptr<SleeveFolder>               _viewing_node;
  sl_path_t                                 _viewing_path;
  std::vector<std::weak_ptr<SleeveElement>> _children;
  std::shared_ptr<ImageLoader>              _loader;

  std::shared_ptr<ImagePoolManager>         _image_pool;
  std::vector<DisplayingImage>              _to_display;

 public:
  SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool);
  SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool, sl_path_t viewing_path);
  SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool,
             std::shared_ptr<SleeveFolder> viewing_node, sl_path_t viewing_path);

  void UpdateView();
  void LoadPreview(uint32_t range_low, uint32_t range_high);
};
};  // namespace puerhlab