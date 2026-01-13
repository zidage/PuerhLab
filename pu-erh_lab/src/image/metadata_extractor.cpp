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

#include "image/metadata_extractor.hpp"

namespace puerhlab {
static void GetDisplayMetadataFromExif(Exiv2::ExifData&     exif_data,
                                       ExifDisplayMetaData& display_metadata) {
  if (exif_data.empty()) {
    return;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Make")) != exif_data.end()) {
    display_metadata.make_ = exif_data["Exif.Image.Make"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Model")) != exif_data.end()) {
    display_metadata.model_ = exif_data["Exif.Image.Model"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensModel")) != exif_data.end()) {
    display_metadata.lens_ = exif_data["Exif.Photo.LensModel"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensMake")) != exif_data.end()) {
    display_metadata.lens_make_ = exif_data["Exif.Photo.LensMake"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FNumber")) != exif_data.end()) {
    auto aperture_rational = exif_data["Exif.Photo.FNumber"].toRational();
    display_metadata.aperture_ =
        static_cast<float>(aperture_rational.first) / aperture_rational.second;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exif_data.end()) {
    auto exposure_rational = exif_data["Exif.Photo.FocalLength"].toRational();
    display_metadata.focal_ =
        static_cast<float>(exposure_rational.first) / exposure_rational.second;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings")) != exif_data.end()) {
    display_metadata.iso_ = exif_data["Exif.Photo.ISOSpeedRatings"].toInt64();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ShutterSpeedValue")) != exif_data.end()) {
    display_metadata.shutter_speed_ = exif_data["Exif.Image.ShutterSpeedValue"].toRational();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageLength")) != exif_data.end()) {
    display_metadata.height_ = exif_data["Exif.Image.ImageLength"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageWidth")) != exif_data.end()) {
    display_metadata.width_ = exif_data["Exif.Image.ImageWidth"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.DateTime")) != exif_data.end()) {
    display_metadata.date_time_str_ = exif_data["Exif.Image.DateTime"].toString();
    if (display_metadata.date_time_str_.size() >= 10) {
      display_metadata.date_time_str_[4] =
          '-';  // Change from "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
      display_metadata.date_time_str_[7] = '-';
    }
  }
}

auto MetadataExtractor::ExtractEXIF(const image_path_t& image_path) -> Exiv2::Image::UniquePtr {
  Exiv2::Image::UniquePtr image = Exiv2::ImageFactory::open(image_path.string());
  image->readMetadata();
  return image;
}

auto MetadataExtractor::EXIFToDisplayMetaData(const Exiv2::Image::UniquePtr& exif_data)
    -> ExifDisplayMetaData {
  ExifDisplayMetaData display_metadata;
  if (exif_data->exifData().empty()) {
    return display_metadata;
  }
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  return display_metadata;
}


auto MetadataExtractor::EXIFToJSON(const Exiv2::Image::UniquePtr& exif_data) -> nlohmann::json {
  // The full EXIF is too large, we only convert the display-friendly metadata to JSON
  nlohmann::json exif_json;
  auto           display_metadata = EXIFToDisplayMetaData(exif_data);
  exif_json                       = display_metadata.ToJson();
  return exif_json;
}

void MetadataExtractor::ExtractEXIF_ToImage(const image_path_t& image_path, Image& image) {
  auto exif_data = ExtractEXIF(image_path);
  if (exif_data->exifData().empty()) {
    return;
  }
  ExifDisplayMetaData display_metadata;
  GetDisplayMetadataFromExif(exif_data->exifData(), display_metadata);
  image.SetExifDisplayMetaData(std::move(display_metadata));
}
}  // namespace puerhlab