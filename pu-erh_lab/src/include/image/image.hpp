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
  /**
   * @brief related edit history of this image
   *
   */
  EditHistory _edit_history;
  /**
   * @brief a pointer to the current edit version
   *
   */
  std::shared_ptr<Version> _curr_version;

 public:
  /**
   * @brief uid of the image
   *
   */
  image_id_t _image_id;
  /**
   * @brief the path to this image
   *
   */
  image_path_t _image_path;
  /**
   * @brief the exif metadata of this image
   *
   */
  Exiv2::ExifData _exif_data;
  /**
   * @brief image data, represented by a opencv image. It is not empty if and only if has_data is true.
   *
   */
  bool    has_data;
  cv::Mat _image_data;
  /**
   * @brief thumbnail of this image
   *
   */
  cv::Mat _thumbnail;
  /**
   * @brief image type
   *
   */
  ImageType _image_type = ImageType::DEFAULT;

  /**
   * @brief Construct a new Image object
   *
   * @param image_id the interal uid given to the new image
   * @param image_path the disk location of the image
   * @param image_type the type of the image
   */
  explicit Image(image_id_t image_id, image_path_t image_path, ImageType image_type);

  /**
   * @brief Load image data into an image object
   *
   * @param image_data
   */
  void LoadData(cv::Mat &&load_image);

  /**
   * @brief Load image thumbnail into an image object
   *
   * @param image_data
   */
  void LoadThumbnail(cv::Mat &&thumbnail);
};
};  // namespace puerhlab
