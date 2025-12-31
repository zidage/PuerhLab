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
#include <exiv2/exif.hpp>
#include <exiv2/image.hpp>
#include <future>
#include <memory>
#include <opencv2/imgcodecs.hpp>

#include "concurrency/thread_pool.hpp"
#include "image/image.hpp"
#include "type/type.hpp"
#include "utils/queue/queue.hpp"

#define MAX_REQUEST_SIZE 64u
namespace puerhlab {

enum class DecodeType { SLEEVE_LOADING, THUMB, RAW, REGULAR };

class DecoderScheduler {
 private:
  ThreadPool                   _file_read_thread_pool;
  ThreadPool                   _thread_pool;
  std::shared_ptr<BufferQueue> _decoded_buffer;

 public:
  explicit DecoderScheduler(size_t thread_count, std::shared_ptr<BufferQueue> decoded_buffer);

  void ScheduleDecode(image_id_t id, image_path_t image_path,
                      std::shared_ptr<std::promise<image_id_t>> decode_promise);

  void ScheduleDecode(std::shared_ptr<Image> source_img, DecodeType decode_type,
                      std::shared_ptr<std::promise<image_id_t>> decode_promise);
};

};  // namespace puerhlab