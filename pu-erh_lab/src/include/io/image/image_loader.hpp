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

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "decoders/decoder_scheduler.hpp"
#include "image/image.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"
namespace puerhlab {

class ImageLoader {
 private:
  // Image decoding part
  std::shared_ptr<BufferQueue>                           buffer_decoded_;
  uint32_t                                               buffer_size_;
  size_t                                                 use_thread_;
  image_id_t                                             start_id_;
  image_id_t                                             next_id_;
  DecoderScheduler                                       decoder_scheduler_;

  std::vector<std::shared_ptr<std::promise<image_id_t>>> promises_;
  std::vector<std::future<image_id_t>>                   futures_;

 public:
  explicit ImageLoader(uint32_t buffer_size, size_t use_thread, image_id_t start_id);

  void StartLoading(std::vector<image_path_t> images, DecodeType decode_type);
  void StartLoading(std::shared_ptr<Image> source_img, DecodeType decode_type);
  auto LoadImage() -> std::shared_ptr<Image>;
};

class ByteBufferLoader {
 public:
  /**
   * @brief Load image data from disk into a byte buffer, used only from ImagePool
   *
   * @param img
   * @return std::shared_ptr<std::vector<uint8_t>>
   */
  static auto LoadFromImage(std::shared_ptr<Image> img) -> std::shared_ptr<std::vector<uint8_t>>;
};
};  // namespace puerhlab