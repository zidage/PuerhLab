/*
 * @file        pu-erh_lab/src/include/decoders/image_decoder.hpp
 * @brief       A decoder responsible for decoding image files
 * @author      Yurun Zi
 * @date        2025-03-28
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

#pragma once

#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"
#include "type/type.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exiv2/exif.hpp>
#include <exiv2/image.hpp>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#define MAX_REQUEST_SIZE 64u
namespace puerhlab {

class ImageDecoder {
private:
  ThreadPool _thread_pool;
  uint32_t _total_request;
  std::atomic<uint32_t> _next_request_id;
  std::atomic<uint32_t> _completed_request;
  std::vector<Image> _decoded;

public:
  explicit ImageDecoder(size_t thread_count, uint32_t total_request);

  void ScheduleDecode(image_path_t image_path);

  
};

static void DecodeImage(std::ifstream &&file, file_path_t file_path,
  std::vector<Image> result, uint32_t id);

}; // namespace puerhlab