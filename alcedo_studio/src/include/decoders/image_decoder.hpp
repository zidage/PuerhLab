//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <exiv2/exif.hpp>
#include <exiv2/image.hpp>
#include <filesystem>
#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "image/image.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

#define MAX_REQUEST_SIZE 64u
namespace alcedo {

class ImageDecoder {
 public:
  virtual void Decode(std::vector<char> buffer, std::filesystem::path file_path,
                      std::shared_ptr<BufferQueue> result, image_id_t id,
                      std::shared_ptr<std::promise<image_id_t>> promise) = 0;
};
};  // namespace alcedo