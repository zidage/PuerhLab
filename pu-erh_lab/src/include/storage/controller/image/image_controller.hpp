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

#include <memory>
#include <string>
#include <vector>

#include "image/image.hpp"
#include "storage/controller/controller_types.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/service/image/image_service.hpp"
#include "type/type.hpp"

namespace puerhlab {
class ImageController {
 private:
  ConnectionGuard guard_;
  ImageService    service_;

 public:
  ImageController(ConnectionGuard&& guard);
  void CaptureImagePool(const std::shared_ptr<ImagePoolManager> image_pool);
  void AddImage(const std::shared_ptr<Image> image);
  void RemoveImageById(const image_id_t remove_id);
  void RemoveImageByType(const ImageType type);
  void RemoveImageByPath(const std::wstring& path);
  void UpdateImage(const image_id_t remove_id);
  auto GetImageById(const image_id_t id) -> std::shared_ptr<Image>;
  auto GetImageByType(const ImageType type) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByName(const std::wstring& name) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByPath(const std::filesystem::path path) -> std::vector<std::shared_ptr<Image>>;
};
};  // namespace puerhlab