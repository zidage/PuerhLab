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
#include <filesystem>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <type/type.hpp>

namespace puerhlab {
enum class ImageType { DEFAULT, JPEG, PNG, TIFF, ARW, CR2, CR3, NEF, DNG };

/**
 * @brief Represent a tracked image file
 *
 */
class Image {
 public:
  image_id_t        _image_id;
  image_path_t      _image_path;
  file_name_t       _image_name;

  Exiv2::ExifData   _exif_data;
  cv::Mat           _image_data;
  cv::Mat           _thumbnail;
  ImageType         _image_type = ImageType::DEFAULT;

  std::atomic<bool> _has_thumbnail;

  p_hash_t          _checksum;

  std::atomic<bool> _has_full_img;
  std::atomic<bool> _has_thumb;
  std::atomic<bool> _has_exif;

  std::atomic<bool> _thumb_pinned = false;
  std::atomic<bool> _full_pinned  = false;

  explicit Image()                = default;
  explicit Image(image_id_t image_id, image_path_t image_path, ImageType image_type, Exiv2::ExifData exif_data);
  explicit Image(image_id_t image_id, image_path_t image_path, file_name_t image_name, ImageType image_type,
                 Exiv2::ExifData exif_data);
  explicit Image(image_path_t image_path, ImageType image_type, Exiv2::ExifData exif_data);
  explicit Image(Image &&other);

  friend std::wostream &operator<<(std::wostream &os, const Image &img);

  void                  LoadData(cv::Mat &&load_image);
  void                  LoadThumbnail(cv::Mat &&thumbnail);
  auto                  GetImageData() -> cv::Mat &;
  void                  SetId(image_id_t image_id);
  void                  ClearData();
  void                  ClearThumbnail();
  auto                  ExifToJson() -> std::string;
};
};  // namespace puerhlab
