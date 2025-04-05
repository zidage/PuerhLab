/*
 * @file        pu-erh_lab/src/include/raw/raw_decoder.hpp
 * @brief       header file for raw decoder module
 * @author      Yurun Zi
 * @date        2025-03-19
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

#include "image_decoder.hpp"
#include "type/type.hpp"

#include <libraw/libraw.h>
#include <memory>

namespace puerhlab {

class RawDecoder : public ImageDecoder {
public:
  RawDecoder() = default;
  void Decode(std::vector<char> buffer, file_path_t file_path,
              NonBlockingQueue<std::optional<Image>> &result, uint32_t id,
              std::shared_ptr<std::promise<uint32_t>> promise);
};

}; // namespace puerhlab
