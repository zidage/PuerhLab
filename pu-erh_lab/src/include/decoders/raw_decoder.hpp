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

#include <libraw/libraw.h>

#include <memory>

#include "data_decoder.hpp"
#include "type/type.hpp"

namespace puerhlab {
enum class OutputColorSpace : int {
  RAW         = 0,
  sRGB        = 1,
  AdobeRGB    = 2,
  Wide        = 3,
  ProPhotoRGB = 4,
  XYZ         = 5,
  ACES        = 6,
  DCIP3       = 7,
  REC2020     = 8
};

class RawDecoder : public DataDecoder {
 public:
  RawDecoder() = default;
  void Decode(std::vector<char> buffer, std::filesystem::path file_path,
              std::shared_ptr<BufferQueue> result, image_id_t id,
              std::shared_ptr<std::promise<image_id_t>> promise);

  void Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
              std::shared_ptr<BufferQueue>              result,
              std::shared_ptr<std::promise<image_id_t>> promise);

  void Decode(std::vector<char>&& buffer, std::shared_ptr<Image> source_img);
};

};  // namespace puerhlab
