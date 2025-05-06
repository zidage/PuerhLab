#include "sleeve/sleeve_view.hpp"

#include <memory>

#include "image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_file.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"

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
  auto elements = _viewing_node.lock()->ListElements();
  for (auto &e : *elements) {
    _children.push_back(_base->AccessElementById(e).value());
  }
}

void SleeveView::LoadPreview() {
  _loader = std::make_shared<ImageLoader>(64, 8, 0);
  for (auto &e : _children) {
    auto e_shared = e.lock();
    if (e_shared->_type == ElementType::FILE) {
      _loader->StartLoading(std::dynamic_pointer_cast<SleeveFile>(e_shared)->GetImage(), DecodeType::THUMB);
    }
  }
}

};  // namespace puerhlab