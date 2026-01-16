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

Image::Image(image_id_t image_id) : image_id_(image_id) {}

/**
 * @brief Construct a new Image object
 *
 * @param image_id the interal uid given to the new image
 * @param image_path the disk location of the image
 * @param image_type the type of the image
 */
Image::Image(image_id_t image_id, image_path_t image_path, ImageType image_type)
    : image_id_(image_id), image_path_(image_path), image_type_(image_type) {}

Image::Image(image_id_t image_id, image_path_t image_path, file_name_t image_name,
             ImageType image_type)
    : image_id_(image_id),
      image_path_(image_path),
      image_name_(image_name),
      image_type_(image_type) {}

Image::Image(image_path_t image_path, ImageType image_type)
    : image_path_(image_path), image_type_(image_type) {}

Image::Image(Image&& other)
    : image_id_(other.image_id_),
      image_path_(std::move(other.image_path_)),
      exif_data_(std::move(other.exif_data_)),
      image_data_(std::move(other.image_data_)),
      thumbnail_(std::move(other.thumbnail_)),
      image_type_(other.image_type_) {}

std::wostream& operator<<(std::wostream& os, const Image& img) {
  os << "img_id: " << img.image_id_ << "\timage_path: " << img.image_path_.wstring()
     << L"\tAdded Time: ";
  return os;
}

/**
 * @brief Load image data into an image object
 *
 * @param image_data
 */
void Image::LoadOriginalData(ImageBuffer&& load_image) {
  image_data_   = std::move(load_image);
  has_full_img_ = true;
}

void Image::LoadThumbnailData(ImageBuffer&& thumbnail) {
  thumbnail_     = std::move(thumbnail);
  has_thumbnail_ = true;
}

void Image::ClearData() {
  image_data_.ReleaseCPUData();
  has_full_img_ = false;
}

void Image::ClearThumbnail() {
  thumbnail_.ReleaseCPUData();
  has_thumbnail_ = false;
}

auto Image::ExifToJson() -> std::string {
  // If the image has exif json, return it directly
  if (has_exif_json_) {
    return nlohmann::to_string(exif_json_);
  }

  exif_json_     = exif_display_.ToJson();
  has_exif_json_ = true;
  return nlohmann::to_string(exif_json_);
}

void Image::JsonToExif(std::string json_str) {
  try {
    exif_json_     = nlohmann::json::parse(json_str);
    has_exif_json_ = true;
    exif_display_.FromJson(exif_json_);
  } catch (nlohmann::json::parse_error& e) {
    throw std::runtime_error("[ERROR] Image: JSON parse error, " + std::string(e.what()));
  } catch (std::exception& e) {
    throw std::runtime_error("[ERROR] Image: JSON to Exif conversion error, " +
                             std::string(e.what()));
  }
}

void Image::SetId(image_id_t image_id) { image_id_ = image_id; }

void Image::SetExifDisplayMetaData(ExifDisplayMetaData&& exif_display) {
  exif_display_     = std::move(exif_display);
  has_exif_display_ = true;
}

void Image::ComputeChecksum() { checksum_ = XXH3_64bits(this, sizeof(*this)); }

auto Image::GetImageData() -> cv::Mat& { return image_data_.GetCPUData(); }

auto Image::GetThumbnailData() -> cv::Mat& { return thumbnail_.GetCPUData(); }

auto Image::GetThumbnailBuffer() -> ImageBuffer& { return thumbnail_; }

void Image::MarkSyncState(ImageSyncState state) { sync_state_ = state; }

auto Image::GetSyncState() -> ImageSyncState { return sync_state_.load(); }
};  // namespace puerhlab
