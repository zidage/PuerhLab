/*
 * @file        pu-erh_lab/src/include/mapper/sleeve/sleeve_element.hpp
 * @brief       A exif-based filter for internal filtering in a sleeve base
 * @author      Yurun Zi
 * @date        2025-03-26
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

#include <cstdint>
#include <ctime>

#include "type/type.hpp"

namespace puerhlab {
enum class ElementType { FILE, FOLDER };

/**
 * @brief Abstract objects residing in a sleeve, it can be files or folders
 *
 */
class SleeveElement {
 protected:
  sl_element_id_t _element_id;
  ElementType     _e_type;

  std::time_t _added_time;
  std::time_t _last_modified_time;

  explicit SleeveElement(sl_element_id_t id);

  void SetAddTime();
  void SetLastModifiedTime();
};
};  // namespace puerhlab
