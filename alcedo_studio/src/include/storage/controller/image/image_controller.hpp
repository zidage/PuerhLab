//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "image/image.hpp"
#include "storage/controller/controller_types.hpp"
#include "storage/image_pool/image_pool_manager.hpp"
#include "storage/service/image/image_service.hpp"
#include "type/type.hpp"

namespace alcedo {
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

  void UpdateImage(const std::shared_ptr<Image> image);
  
  auto GetImageById(const image_id_t id) -> std::shared_ptr<Image>;
  auto GetImageByType(const ImageType type) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByName(const std::wstring& name) -> std::vector<std::shared_ptr<Image>>;
  auto GetImageByPath(const std::filesystem::path path) -> std::vector<std::shared_ptr<Image>>;
};
};  // namespace alcedo