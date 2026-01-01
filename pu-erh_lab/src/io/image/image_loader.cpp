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

#include "io/image/image_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <future>
#include <memory>
#include <vector>

#include "decoders/decoder_scheduler.hpp"
#include "type/supported_file_type.hpp"
#include "type/type.hpp"

namespace puerhlab {
/**
 * @brief Construct a new ImageLoader::ImageLoader object
 *
 * @param buffer_size size of loader's buffer
 * @param use_thread number of thread used to decode image
 */
ImageLoader::ImageLoader(uint32_t buffer_size, size_t use_thread, image_id_t start_id)
    : buffer_decoded_(std::make_shared<BufferQueue>(buffer_size)),
      buffer_size_(buffer_size),
      use_thread_(use_thread),
      start_id_(start_id),
      next_id_(start_id),
      decoder_scheduler_(use_thread, buffer_decoded_) {}

/**
 * @brief Loads a batch of images
 *
 * @param images
 * @param decode_type
 */
void ImageLoader::StartLoading(std::vector<image_path_t> images, DecodeType decode_type) {
  for (const auto& img : images) {
    // Skip unsupported file type
    // if (!is_supported_file(img)) {
    //   continue;
    // }

    promises_.emplace_back(std::make_shared<std::promise<image_id_t>>());
    futures_.emplace_back(promises_[next_id_]->get_future());
    if (decode_type == DecodeType::SLEEVE_LOADING)
      decoder_scheduler_.ScheduleDecode(next_id_, img, promises_[next_id_]);
    ++next_id_;
  }
}

/**
 * @brief Loads a single of images
 *
 * @param images
 * @param decode_type
 */
void ImageLoader::StartLoading(std::shared_ptr<Image> image, DecodeType decode_type) {
  promises_.emplace_back(std::make_shared<std::promise<image_id_t>>());
  futures_.emplace_back(promises_[next_id_]->get_future());
  decoder_scheduler_.ScheduleDecode(image, decode_type, promises_[next_id_]);
  ++next_id_;
}

auto ImageLoader::LoadImage() -> std::shared_ptr<Image> {
  // If there's no finished image in the buffer, will block the load routine
  std::shared_ptr<Image> img = buffer_decoded_->pop();
  return img;
}

auto ByteBufferLoader::LoadFromImage(std::shared_ptr<Image> img)
    -> std::shared_ptr<std::vector<uint8_t>> {
  std::ifstream   file(img->image_path_, std::ios::binary | std::ios::ate);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  auto buffer = std::make_shared<std::vector<uint8_t>>(fileSize);
  if (!file.read(reinterpret_cast<char*>(buffer->data()), fileSize)) {
    return nullptr;
  }
  file.close();
  return buffer;
}

};  // namespace puerhlab