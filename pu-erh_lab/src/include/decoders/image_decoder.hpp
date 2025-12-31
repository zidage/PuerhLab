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
namespace puerhlab {

class ImageDecoder {
 public:
  virtual void Decode(std::vector<char> buffer, std::filesystem::path file_path,
                      std::shared_ptr<BufferQueue> result, image_id_t id,
                      std::shared_ptr<std::promise<image_id_t>> promise) = 0;
};
};  // namespace puerhlab