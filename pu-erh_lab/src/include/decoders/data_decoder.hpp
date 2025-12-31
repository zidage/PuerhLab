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

#include "image_decoder.hpp"

namespace puerhlab {
class DataDecoder : public ImageDecoder {
 public:
  virtual void Decode(std::vector<char> buffer, std::filesystem::path file_path,
                      std::shared_ptr<BufferQueue> result, image_id_t id,
                      std::shared_ptr<std::promise<image_id_t>> promise) = 0;

  virtual void Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                      std::shared_ptr<BufferQueue>              result,
                      std::shared_ptr<std::promise<image_id_t>> promise) = 0;
};

};  // namespace puerhlab