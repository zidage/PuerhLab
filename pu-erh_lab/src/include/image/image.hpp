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

#include <exiv2/exif.hpp>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <json.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <type/type.hpp>

#include "image/image_buffer.hpp"
#include "image/metadata.hpp"

namespace puerhlab {
enum class ImageType { DEFAULT, JPEG, PNG, TIFF, ARW, CR2, CR3, NEF, DNG };

/**
 * @brief Represent a tracked image file
 *
 */
class Image {
 public:
  image_id_t              _image_id;
  image_path_t            _image_path;
  file_name_t             _image_name;

  Exiv2::Image::UniquePtr _exif_data;
  nlohmann::json          _exif_json;
  ExifDisplayMetaData     _exif_display;

  ImageBuffer             _image_data;
  ImageBuffer             _thumbnail;
  ImageType               _image_type = ImageType::DEFAULT;

  std::atomic<bool>       _has_thumbnail;

  p_hash_t                _checksum;

  std::atomic<bool>       _has_full_img;
  std::atomic<bool>       _has_thumb;
  std::atomic<bool>       _has_exif;
  std::atomic<bool>       _has_exif_json;
  std::atomic<bool>       _has_exif_display;

  std::atomic<bool>       _thumb_pinned = false;
  std::atomic<bool>       _full_pinned  = false;

  explicit Image()                      = default;
  explicit Image(image_id_t image_id, image_path_t image_path, ImageType image_type);
  explicit Image(image_id_t image_id, image_path_t image_path, file_name_t image_name,
                 ImageType image_type);
  explicit Image(image_path_t image_path, ImageType image_type);
  explicit Image(Image&& other);

  friend std::wostream& operator<<(std::wostream& os, const Image& img);

  void                  LoadData(ImageBuffer&& load_image);
  void                  LoadThumbnail(ImageBuffer&& thumbnail);
  auto                  GetImageData() -> cv::Mat&;
  auto                  GetThumbnailData() -> cv::Mat&;
  auto                  GetThumbnailBuffer() -> ImageBuffer&;
  void                  SetId(image_id_t image_id);
  void                  ClearData();
  void                  ClearThumbnail();
  void                  ComputeChecksum();
  auto                  ExifToJson() -> std::string;
  void                  JsonToExif(std::string json_str);
};
};  // namespace puerhlab
