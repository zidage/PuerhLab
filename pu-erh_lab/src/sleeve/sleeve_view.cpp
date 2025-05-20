#include "sleeve/sleeve_view.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {

DisplayingImage::DisplayingImage(std::weak_ptr<Image> displaying, bool _require_thumb, bool _require_full) {
  _displaying                = displaying.lock();
  _displaying->_full_pinned  = _require_full;
  _displaying->_thumb_pinned = _require_thumb;
}

DisplayingImage::DisplayingImage(std::shared_ptr<Image> displaying, bool _require_thumb, bool _require_full) {
  _displaying                = displaying;
  _displaying->_full_pinned  = _require_full;
  _displaying->_thumb_pinned = _require_thumb;
}

DisplayingImage::~DisplayingImage() {
  _displaying->_full_pinned  = false;
  _displaying->_thumb_pinned = false;
}

SleeveView::SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool)
    : _base(base), _image_pool(image_pool), _loader(64, 24, 0) {}

SleeveView::SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool,
                       sl_path_t viewing_path)
    : _base(base), _viewing_path(viewing_path), _image_pool(image_pool), _loader(64, 24, 0) {
  UpdateView();
}

SleeveView::SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<ImagePoolManager> image_pool,
                       std::shared_ptr<SleeveFolder> viewing_node, sl_path_t viewing_path)
    : _base(base),
      _viewing_node(viewing_node),
      _viewing_path(viewing_path),
      _image_pool(image_pool),
      _loader(64, 24, 0) {
  auto elements = _viewing_node.lock()->ListElements();
  for (auto &e : *elements) {
    _children.push_back(_base->AccessElementById(e).value());
  }
}

void SleeveView::UpdateView() {
  auto target = _base->GetReadGuard(_viewing_path);
  if (!target.has_value() || target.value()._access_element->_type != ElementType::FOLDER) {
    return;
  }
  _viewing_node = std::dynamic_pointer_cast<SleeveFolder>(target->_access_element);
  _children.clear();
  auto elements = _viewing_node.lock()->ListElements();
  for (auto &e : *elements) {
    _children.push_back(_base->AccessElementById(e).value());
  }
}

void SleeveView::UpdateView(sl_path_t new_viewing_path) {
  _viewing_path = new_viewing_path;
  UpdateView();
}

void SleeveView::LoadPreview(uint32_t range_low, uint32_t range_high,
                             std::function<void(size_t, std::weak_ptr<Image>)> callback) {
  std::unordered_map<image_id_t, size_t> index_map;
  uint32_t                               empty_img_count = 0;

  // to_display is a array to store images that require thumbnail lock.
  // once LoadPreview returns, all the images in the to_display will be released from their thumbnail locks
  std::vector<DisplayingImage>           to_display;
  range_high = range_high > _children.size() - 1 ? _children.size() - 1 : range_high;
  for (size_t i = range_low; i <= range_high; ++i) {
    auto e_shared = _children[i].lock();
    if (e_shared->_type == ElementType::FILE) {
      auto e_file  = std::dynamic_pointer_cast<SleeveFile>(e_shared);
      auto img_opt = _image_pool->AccessElement(e_file->GetImage()->_image_id, AccessType::THUMB);
      if (!img_opt.has_value()) {
        _loader.StartLoading(e_file->GetImage(), DecodeType::THUMB);
        ++empty_img_count;
      } else {
        // TODO: notify the UI framework in advance
        callback(i, img_opt.value());
        to_display.push_back({img_opt.value(), true, false});
      }
      index_map[e_file->GetImage()->_image_id] = i;
    } else {
      // Notify UI to display the "folder icon"
    }
  }

  // The size of the thumbnail cache should be slightly bigger than the window size
  // TODO: For now, the cache size can only be expanded. It is okay, since the sleeve view will flush the cache
  // periodically
  if (_image_pool->Capacity(AccessType::THUMB) < range_high - range_low + 10)
    _image_pool->ResizeCache(range_high - range_low + 10, AccessType::THUMB);
  // TODO: Fetch the loaded image, notify UI to update
  for (size_t i = 0; i < empty_img_count; i++) {
    auto   loaded     = _loader.LoadImage();
    size_t view_index = index_map.at(loaded->_image_id);
    // Do something with view_index...
    callback(view_index, loaded);
    _image_pool->RecordAccess(loaded->_image_id, AccessType::THUMB);
    to_display.push_back({loaded, true, false});
  }
}

};  // namespace puerhlab