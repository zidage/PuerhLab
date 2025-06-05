/*
 * @file        pu-erh_lab/src/include/sleeve/sleeve_element/sleeve_file.hpp
 * @brief       A subtype of sleeve element representing a pure file in a typical file system
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
#include <memory>

#include "edit/history/edit_history.hpp"
#include "edit/history/version.hpp"
#include "image/image.hpp"
#include "sleeve_element.hpp"
#include "type/type.hpp"

namespace puerhlab {

/**
 * @brief A type of element, it contains an image file, its edit history, and other metadata used in
 * this software
 *
 */
class SleeveFile : public SleeveElement {
 private:
  std::shared_ptr<Image>       _image;

  std::shared_ptr<EditHistory> _edit_history;
  std::shared_ptr<Version>     _current_version;

 public:
  image_id_t _image_id;
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name);
  explicit SleeveFile(sl_element_id_t id, file_name_t element_name, std::shared_ptr<Image> image);

  auto Clear() -> bool;

  auto Copy(sl_element_id_t new_id) const -> std::shared_ptr<SleeveElement>;
  auto GetImage() -> std::shared_ptr<Image>;
  void SetImage(const std::shared_ptr<Image> img);
  ~SleeveFile();
};
};  // namespace puerhlab