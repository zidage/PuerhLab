/*
 * @file        pu-erh_lab/src/image/image_loader.cpp
 * @brief       A module to load images into memory
 * @author      Yurun Zi
 * @date        2025-04-06
 * @license     MIT
 *
 * @copyright   Copyright (c) 2025 Yurun Zi
 */

// Copyright (c) 2025 Yurun Zi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    : _buffer_decoded(std::make_shared<BufferQueue>(buffer_size)),
      _buffer_size(buffer_size),
      _use_thread(use_thread),
      _start_id(start_id),
      _next_id(start_id),
      _decoder_scheduler(use_thread, _buffer_decoded) {}

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

    promises.emplace_back(std::make_shared<std::promise<image_id_t>>());
    futures.emplace_back(promises[_next_id]->get_future());
    if (decode_type == DecodeType::SLEEVE_LOADING)
      _decoder_scheduler.ScheduleDecode(_next_id, img, promises[_next_id]);
    ++_next_id;
  }
}

/**
 * @brief Loads a single of images
 *
 * @param images
 * @param decode_type
 */
void ImageLoader::StartLoading(std::shared_ptr<Image> image, DecodeType decode_type) {
  promises.emplace_back(std::make_shared<std::promise<image_id_t>>());
  futures.emplace_back(promises[_next_id]->get_future());
  _decoder_scheduler.ScheduleDecode(image, decode_type, promises[_next_id]);
  ++_next_id;
}

auto ImageLoader::LoadImage() -> std::shared_ptr<Image> {
  // If there's no finished image in the buffer, will block the load routine
  std::shared_ptr<Image> img = _buffer_decoded->pop();
  return img;
}

auto ByteBufferLoader::LoadFromImage(std::shared_ptr<Image> img) -> std::vector<uint8_t> {
  std::ifstream   file(img->_image_path, std::ios::binary | std::ios::ate);
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    return {};
  }
  file.close();
  return std::vector<uint8_t>(buffer.begin(), buffer.end());
}

};  // namespace puerhlab