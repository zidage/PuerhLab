//  Copyright 2026 Yurun Zi
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
#include <json.hpp>

#include "image.hpp"
#include "type/type.hpp"

namespace puerhlab {
class MetadataExtractor {
 public:
  /**
   * @brief Extract EXIF metadata from image file
   *
   * @param image_path
   * @return Exiv2::Image::UniquePtr
   */
  static auto ExtractEXIF(const image_path_t& image_path) -> Exiv2::Image::UniquePtr;

  /**
   * @brief Convert EXIF data to JSON format
   *
   * @param exif_data
   * @return nlohmann::json
   */
  static auto EXIFToJSON(const Exiv2::Image::UniquePtr& exif_data) -> nlohmann::json;

  /**
   * @brief Convert EXIF data to display-friendly format
   *
   * @param exif_data
   * @return ExifDisplayMetaData
   */
  static auto EXIFToDisplayMetaData(const Exiv2::Image::UniquePtr& exif_data)
      -> ExifDisplayMetaData;

  /**
   * @brief Extract EXIF metadata and populate the Image object
   *
   * @param image_path
   * @param image
   */
  static void ExtractEXIF_ToImage(const image_path_t& image_path, Image& image);
};
}  // namespace puerhlab