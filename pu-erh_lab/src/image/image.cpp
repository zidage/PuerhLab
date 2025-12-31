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

#include "image/image.hpp"

#include <xxhash.h>

#include <cstdint>
#include <exception>
#include <exiv2/exif.hpp>
#include <exiv2/tags.hpp>
#include <json.hpp>
#include <stdexcept>
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

Image::Image(image_id_t image_id, image_path_t image_path, file_name_t image_name,
             ImageType image_type)
    : _image_id(image_id),
      _image_path(image_path),
      _image_name(image_name),
      _image_type(image_type) {}

Image::Image(image_path_t image_path, ImageType image_type)
    : _image_path(image_path), _image_type(image_type) {}

Image::Image(Image&& other)
    : _image_id(other._image_id),
      _image_path(std::move(other._image_path)),
      _exif_data(std::move(other._exif_data)),
      _image_data(std::move(other._image_data)),
      _thumbnail(std::move(other._thumbnail)),
      _image_type(other._image_type) {}

std::wostream& operator<<(std::wostream& os, const Image& img) {
  os << "img_id: " << img._image_id << "\timage_path: " << img._image_path.wstring()
     << L"\tAdded Time: ";
  return os;
}

/**
 * @brief Load image data into an image object
 *
 * @param image_data
 */
void Image::LoadData(ImageBuffer&& load_image) {
  _image_data   = std::move(load_image);
  _has_full_img = true;
}

void Image::LoadThumbnail(ImageBuffer&& thumbnail) {
  _thumbnail     = std::move(thumbnail);
  _has_thumbnail = true;
}

void Image::ClearData() {
  _image_data.ReleaseCPUData();
  _has_full_img = false;
}

void Image::ClearThumbnail() {
  _thumbnail.ReleaseCPUData();
  _has_thumbnail = false;
}

auto Image::ExifToJson() -> std::string {
  // If the image has exif json, return it directly
  if (_has_exif_json) {
    return nlohmann::to_string(_exif_json);
  }

  _exif_json     = _exif_display.ToJson();
  _has_exif_json = true;
  return nlohmann::to_string(_exif_json);
}

void Image::JsonToExif(std::string json_str) {
  try {
    _exif_json     = nlohmann::json::parse(json_str);
    _has_exif_json = true;
    _exif_display.FromJson(_exif_json);
  } catch (nlohmann::json::parse_error& e) {
    throw std::runtime_error("Image: JSON parse error");
  }
}

void Image::ComputeChecksum() { _checksum = XXH3_64bits(this, sizeof(*this)); }

auto Image::GetImageData() -> cv::Mat& { return _image_data.GetCPUData(); }

auto Image::GetThumbnailData() -> cv::Mat& { return _thumbnail.GetCPUData(); }

auto Image::GetThumbnailBuffer() -> ImageBuffer& { return _thumbnail; }
};  // namespace puerhlab
