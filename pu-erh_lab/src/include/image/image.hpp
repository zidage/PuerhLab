/*
 * @file        pu-erh_lab/src/include/image/image.hpp
 * @brief       parent class of all image clasess, e.g. tiff, jpeg, raw, etc.
 * @author      Yurun Zi
 * @date        2025-03-20
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

#include <exiv2/exif.hpp>
#include <exiv2/exiv2.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <type/type.hpp>

#include "edit/history/edit_history.hpp"
#include "edit/history/version.hpp"

namespace puerhlab {
enum class ImageType { DEFAULT, JPEG, TIFF, ARW, CR2, CR3, NEF, DNG };

class Image {
 private:
  EditHistory _edit_history;
  std::shared_ptr<Version> _curr_version;
 public:
  image_path_t _image_path;
  // TODO: Add Implementation
  Exiv2::ExifData _exif_data;
};
};  // namespace puerhlab
