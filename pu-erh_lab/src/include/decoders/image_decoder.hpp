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