//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "io/image/image_loader.hpp"
#include "sleeve/sleeve_element/sleeve_element.hpp"
#include "sleeve/sleeve_element/sleeve_folder.hpp"
#include "sleeve/sleeve_filesystem.hpp"
#include "sleeve/sleeve_filter/filter_combo.hpp"
#include "sleeve_base.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "type/type.hpp"

namespace puerhlab {

struct DisplayingImage {
 private:
  std::shared_ptr<Image> displaying_;
  bool                   require_thumb_;
  bool                   require_full_;

 public:
  DisplayingImage(std::weak_ptr<Image> displaying, bool require_thumb, bool require_full);
  DisplayingImage(std::shared_ptr<Image> displaying, bool require_thumb, bool require_full);
  ~DisplayingImage();
};

class SleeveView {
 private:
  std::shared_ptr<FileSystem>               fs_;
  std::weak_ptr<SleeveFolder>               viewing_node_;
  sl_path_t                                 viewing_path_;
  std::vector<std::weak_ptr<SleeveElement>> children_;

  std::shared_ptr<ImagePoolManager>         image_pool_;

  ImageLoader                               loader_;

 public:
  SleeveView(std::shared_ptr<FileSystem> base, std::shared_ptr<ImagePoolManager> image_pool);
  SleeveView(std::shared_ptr<FileSystem> base, std::shared_ptr<ImagePoolManager> image_pool,
             sl_path_t viewing_path);
  SleeveView(std::shared_ptr<FileSystem> base, std::shared_ptr<ImagePoolManager> image_pool,
             std::shared_ptr<SleeveFolder> viewing_node, sl_path_t viewing_path);

  void UpdateView();
  void UpdateView(sl_path_t new_viewing_path);
  void LoadPreview(uint32_t range_low, uint32_t range_high,
                   std::function<void(size_t, std::weak_ptr<Image>)> callback);
};
};  // namespace puerhlab