/*
 * @file        pu-erh_lab/src/image/image.cpp
 * @brief       specify the behaviors of a image
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

#include "image/image.hpp"

#include <exception>
#include <exiv2/exif.hpp>
#include <exiv2/tags.hpp>
#include <json.hpp>
#include <string>
#include <utility>

namespace puerhlab {
using json = nlohmann::json;
/**
 * @brief Construct a new Image object
 *
 * @param image_id the interal uid given to the new image
 * @param image_path the disk location of the image
 * @param image_type the type of the image
 */
Image::Image(image_id_t image_id, image_path_t image_path, ImageType image_type)
    : _image_id(image_id), _image_path(image_path), _image_type(image_type) {}

Image::Image(image_id_t image_id, image_path_t image_path, file_name_t image_name, ImageType image_type)
    : _image_id(image_id), _image_path(image_path), _image_name(image_name), _image_type(image_type) {}

Image::Image(image_path_t image_path, ImageType image_type) : _image_path(image_path), _image_type(image_type) {}

Image::Image(Image &&other)
    : _image_id(other._image_id),
      _image_path(std::move(other._image_path)),
      _exif_data(std::move(other._exif_data)),
      _image_data(std::move(other._image_data)),
      _thumbnail(std::move(other._thumbnail)),
      _image_type(other._image_type) {}

std::wostream &operator<<(std::wostream &os, const Image &img) {
  os << "img_id: " << img._image_id << "\timage_path: " << img._image_path.wstring() << L"\tAdded Time: ";
  return os;
}

/**
 * @brief Load image data into an image object
 *
 * @param image_data
 */
void Image::LoadData(cv::Mat &&load_image) {
  _image_data   = std::move(load_image);
  _has_full_img = true;
}

void Image::LoadThumbnail(cv::Mat &&thumbnail) {
  _thumbnail     = std::move(thumbnail);
  _has_thumbnail = true;
}

void Image::ClearData() {
  _image_data.release();
  _has_full_img = false;
}

void Image::ClearThumbnail() {
  _thumbnail.release();
  _has_thumbnail = false;
}

auto Image::ExifToJson() -> std::string {
  json o;
  // Temporary remove the support for reading exif
  if (!_has_exif) {
    return nlohmann::to_string(o);
  }
  try {
    auto exif_data = _exif_data->exifData();
    for (auto i = exif_data.begin(); i != exif_data.end(); ++i) {
      auto type_name  = i->typeName();          // debug
      auto type_value = i->value().toString();  // debug
      o[type_name]    = type_value;
    }
  } catch (std::exception &e) {
  }
  return nlohmann::to_string(o);
}
};  // namespace puerhlab
