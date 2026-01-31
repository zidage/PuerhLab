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

#include <cstdint>

namespace puerhlab {

// Wide character path c string
#define image_path_t    std::filesystem::path
#define file_path_t     std::filesystem::path
#define file_name_t     std::wstring

#define sl_path_t       std::wstring

#define image_id_t      uint32_t

// Used in sleeve
#define sleeve_id_t     uint32_t
#define sl_element_id_t uint32_t

// Used in filters
#define filter_id_t     uint32_t

// Hash type for version control
#define p_hash_t        uint64_t

// Used in buffer-like structures to represent frame id
#define frame_id_t      uint32_t

// Used in DecodeRequest type
#define request_id_t    size_t

#define BufferQueue     ConcurrentBlockingQueue<std::shared_ptr<Image>>

#define PriorityLevel   int

// #define version_id_t   XXH128_hash_t

enum class DecodeRes {
  FULL,
  HALF,
  QUARTER,
};


};  // namespace puerhlab