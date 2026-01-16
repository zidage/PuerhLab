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
enum class ThumbState : uint8_t { NOT_PRESENT = 0, PENDING, READY, FAILED };
enum class ImageSyncState : uint8_t { SYNCED, UNSYNCED, MODIFIED, DELETED };

/**
 * @brief Represent a tracked image file
 *
 */
class Image {
 public:
  image_id_t                  image_id_;
  image_path_t                image_path_;
  file_name_t                 image_name_;

  Exiv2::Image::UniquePtr     exif_data_;
  nlohmann::json              exif_json_;
  ExifDisplayMetaData         exif_display_;

  ImageBuffer                 image_data_;
  ImageBuffer                 thumbnail_;
  ImageType                   image_type_ = ImageType::DEFAULT;

  std::atomic<bool>           has_thumbnail_;

  std::atomic<ThumbState>     thumb_state_ = ThumbState::NOT_PRESENT;

  std::atomic<ImageSyncState> sync_state_  = ImageSyncState::SYNCED;

  p_hash_t                    checksum_;

  std::atomic<bool>           has_full_img_;
  std::atomic<bool>           has_thumb_;
  std::atomic<bool>           has_exif_;
  std::atomic<bool>           has_exif_json_;
  std::atomic<bool>           has_exif_display_;

  std::atomic<bool>           thumb_pinned_ = false;
  std::atomic<bool>           full_pinned_  = false;

  explicit Image()                          = default;
  explicit Image(image_id_t image_id);
  explicit Image(image_id_t image_id, image_path_t image_path, ImageType image_type);
  explicit Image(image_id_t image_id, image_path_t image_path, file_name_t image_name,
                 ImageType image_type);
  explicit Image(image_path_t image_path, ImageType image_type);
  explicit Image(Image&& other);

  friend std::wostream& operator<<(std::wostream& os, const Image& img);

  void                  LoadOriginalData(ImageBuffer&& load_image);
  void                  LoadThumbnailData(ImageBuffer&& thumbnail);

  auto                  GetImageData() -> cv::Mat&;
  auto                  GetThumbnailData() -> cv::Mat&;
  auto                  GetThumbnailBuffer() -> ImageBuffer&;

  void                  SetId(image_id_t image_id);
  void                  SetExifDisplayMetaData(ExifDisplayMetaData&& exif_display);

  void                  ClearData();
  void                  ClearThumbnail();
  void                  ComputeChecksum();
  auto                  ExifToJson() -> std::string;
  void                  JsonToExif(std::string json_str);

  void                  MarkSyncState(ImageSyncState state);
  auto                  GetSyncState() -> ImageSyncState;
};
};  // namespace puerhlab
