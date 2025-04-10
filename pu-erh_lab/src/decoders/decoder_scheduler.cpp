/*
 * @file        pu-erh_lab/src/include/decoders/decoder_scheduler.hpp
 * @brief       A scheduler to manage image decoding
 * @author      Yurun Zi
 * @date        2025-04-05
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

#include "decoders/decoder_scheduler.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "decoders/image_decoder.hpp"
#include "decoders/metadata_decoder.hpp"
#include "decoders/raw_decoder.hpp"
#include "decoders/thumbnail_decoder.hpp"
#include "image/image.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

namespace puerhlab {
/**
 * @brief Construct a new Image Decoder::Image Decoder object
 *
 * @param thread_count
 * @param total_request
 */
DecoderScheduler::DecoderScheduler(size_t thread_count, std::shared_ptr<BufferQueue> decoded_buffer)
    : _thread_pool(thread_count), _decoded_buffer(decoded_buffer) {}

/**
 * @brief Schedule a decode task for initialize image data. The decode type can only be SLEEVE_LOADING, therefore
 *        decode_type field is omitted.
 *
 * @param image_path the path of the file to be decoded
 * @param decode_promise the corresponding promise to be collected
 */
void DecoderScheduler::ScheduleDecode(image_id_t id, image_path_t image_path,
                                      std::shared_ptr<std::promise<image_id_t>> decode_promise) {
  // Open file as an ifstream

  std::ifstream file(image_path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    // Check whether file openned successfully
    decode_promise->set_exception(
        std::make_exception_ptr(std::runtime_error("File not exists or no read permission.")));
    return;
  }

  // Assign a decoder for the task
  std::shared_ptr<LoadingDecoder> decoder = std::make_shared<MetadataDecoder>();

  // Read file into memory
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    decode_promise->set_exception(
        std::make_exception_ptr(std::runtime_error("File not exists or no read permission.")));
    return;
  }
  file.close();

  // Submit a new decode request
  auto                 &decoded_buffer = _decoded_buffer;
  std::filesystem::path file_path(image_path);

  auto task = std::make_shared<std::packaged_task<void()>>(
      [decoder, buffer = std::move(buffer), file_path, decoded_buffer, id, decode_promise]() mutable {
        decoder->Decode(std::move(buffer), file_path, decoded_buffer, id, decode_promise);
      });

  _thread_pool.Submit([task]() { (*task)(); });
}

/**
 * @brief Schedule a decode task for loading image data into an Image object
 *
 * @param source_img
 * @param decode_promise
 */
void DecoderScheduler::ScheduleDecode(std::shared_ptr<Image> source_img, DecodeType decode_type,
                                      std::shared_ptr<std::promise<image_id_t>> decode_promise) {
  // Open file as an ifstream
  std::ifstream file(source_img->_image_path, std::ios::binary | std::ios::ate);

  if (!file.is_open()) {
    // Check file status
    decode_promise->set_exception(
        std::make_exception_ptr(std::runtime_error("File not exists or no read permission.")));
    return;
  }

  // Assign a decoder for the task
  std::shared_ptr<DataDecoder> decoder;

  // Assign a decoder according to the decode type
  switch (decode_type) {
    case DecodeType::THUMB:
      decoder = std::make_shared<ThumbnailDecoder>();
      break;
    case DecodeType::RAW:
      decoder = std::make_shared<RawDecoder>();
      break;
    case DecodeType::REGULAR:
      // FIXME: Add RegularDecoder
      decoder = std::make_shared<ThumbnailDecoder>();
      break;
    default:
      throw std::runtime_error("Incompatible decode type.");
  }

  // Read file into memory
  std::streamsize fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(fileSize);
  if (!file.read(buffer.data(), fileSize)) {
    decode_promise->set_exception(
        std::make_exception_ptr(std::runtime_error("File not exists or no read permission.")));
    return;
  }
  file.close();

  // Submit a new decode request
  auto                 &decoded_buffer = _decoded_buffer;
  std::filesystem::path file_path(source_img->_image_path);

  auto task = std::make_shared<std::packaged_task<void()>>(
      [decoder, buffer = std::move(buffer), decoded_buffer, &source_img, decode_promise]() mutable {
        decoder->Decode(std::move(buffer), source_img, decoded_buffer, decode_promise);
      });

  _thread_pool.Submit([task]() { (*task)(); });
}
};  // namespace puerhlab