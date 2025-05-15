#include "sleeve/sleeve_view.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "type/type.hpp"

namespace puerhlab {
SleeveView::SleeveView(std::shared_ptr<SleeveBase> base) : _base(base) {}

SleeveView::SleeveView(std::shared_ptr<SleeveBase> base, sl_path_t viewing_path)
    : _base(base), _viewing_path(viewing_path) {
  UpdateView();
}

SleeveView::SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<SleeveFolder> viewing_node,
                       sl_path_t viewing_path)
    : _base(base), _viewing_node(viewing_node), _viewing_path(viewing_path) {
  auto elements = _viewing_node.lock()->ListElements();
  for (auto &e : *elements) {
    _children.push_back(_base->AccessElementById(e).value());
  }
}

void SleeveView::UpdateView() {
  auto target = _base->GetWriteGuard(_viewing_path);
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

void SleeveView::LoadPreview(uint32_t range_low, uint32_t range_high) {
  _loader = std::make_shared<ImageLoader>(64, 8, 0);
  std::unordered_map<image_id_t, size_t> index_map;
  for (size_t i = range_low; i <= range_high; ++i) {
    auto e_shared = _children[i].lock();
    if (e_shared->_type == ElementType::FILE) {
      auto e_file = std::dynamic_pointer_cast<SleeveFile>(e_shared);
      _loader->StartLoading(e_file->GetImage(), DecodeType::THUMB);
      index_map[e_file->GetImage()->_image_id] = i;
    }
  }
  // TODO: Fetch the loaded image, notify UI to update
  for (size_t i = range_low; i <= range_high; i++) {
    auto   loaded     = _loader->LoadImage();
    size_t view_index = index_map.at(loaded->_image_id);
    // Do something with view_index...
  }
}

};  // namespace puerhlab