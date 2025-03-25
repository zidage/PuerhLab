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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "utils/queue/queue.hpp"
#include "type/type.hpp"
#include "concurrency/thread_pool.hpp"

#include <libraw/libraw.h>

#include <atomic>
#include <cstddef>
#include <memory>




#define MAX_REQUEST_SIZE 64
namespace puerhlab {

struct DecodeRequest {
  request_id_t _request_id;
  image_path_t _raw_file_path;
  std::shared_ptr<LibRaw> _image_processor;
};

class RawDecoder {
 private:
  NonBlockingQueue<DecodeRequest> _process_queue;
  std::atomic<size_t> _next_request_id;


 public:
  RawDecoder() = default;

  int NewRequest();

};

};  // namespace puerhlab
