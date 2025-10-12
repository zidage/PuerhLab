/*
 * @file        pu-erh_lab/src/include/type/type.hpp
 * @brief       collection of wrapper types
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
enum class ColorSpace { SRGB, ADOBE_RGB, ACEScc, Camera };
};  // namespace puerhlab