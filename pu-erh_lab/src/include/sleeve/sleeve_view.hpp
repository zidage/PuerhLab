#pragma once

#include <memory>
#include <vector>

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
  std::shared_ptr<SleeveFolder>             _viewing_node;
  sl_path_t                                 _viewing_path;
  std::vector<std::weak_ptr<SleeveElement>> _children;

 public:
  SleeveView(std::shared_ptr<SleeveBase> base);
  SleeveView(std::shared_ptr<SleeveBase> base, sl_path_t viewing_path);
  SleeveView(std::shared_ptr<SleeveBase> base, std::shared_ptr<SleeveFolder> viewing_node, sl_path_t viewing_path);

  void UpdateView();
};
};  // namespace puerhlab