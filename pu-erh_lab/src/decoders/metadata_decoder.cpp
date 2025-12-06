/*
 * @file        pu-erh_lab/src/include/decoders/metadata_decoder.hpp
 * @brief       A decoder used to read metadata in a image file, no image data
 * will be loaded.
 * @author      Yurun Zi
 * @date        2025-04-08
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

#include "decoders/metadata_decoder.hpp"

#include <libraw/libraw.h>

#include <chrono>
#include <exiv2/basicio.hpp>
#include <exiv2/exif.hpp>
#include <exiv2/image.hpp>
#include <exiv2/tags.hpp>
#include <exiv2/types.hpp>
#include <filesystem>
#include <functional>
#include <stdexcept>

#include "type/type.hpp"

namespace puerhlab {
void GetDisplayMetadataFromExif(Exiv2::ExifData& exif_data, ExifDisplayMetaData& display_metadata) {
  if (exif_data.empty()) {
    return;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Make")) != exif_data.end()) {
    display_metadata.make = exif_data["Exif.Image.Make"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.Model")) != exif_data.end()) {
    display_metadata.model = exif_data["Exif.Image.Model"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensModel")) != exif_data.end()) {
    display_metadata.lens = exif_data["Exif.Photo.LensModel"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.LensMake")) != exif_data.end()) {
    display_metadata.lens_make = exif_data["Exif.Photo.LensMake"].toString();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FNumber")) != exif_data.end()) {
    auto aperture_rational = exif_data["Exif.Photo.FNumber"].toRational();
    display_metadata.aperture =
        static_cast<float>(aperture_rational.first) / aperture_rational.second;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exif_data.end()) {
    auto exposure_rational = exif_data["Exif.Photo.FocalLength"].toRational();
    display_metadata.focal = static_cast<float>(exposure_rational.first) / exposure_rational.second;
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Photo.ISOSpeedRatings")) != exif_data.end()) {
    display_metadata.iso = exif_data["Exif.Photo.ISOSpeedRatings"].toInt64();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ShutterSpeedValue")) != exif_data.end()) {
    display_metadata.shutter_speed = exif_data["Exif.Image.ShutterSpeedValue"].toRational();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageLength")) != exif_data.end()) {
    display_metadata.height = exif_data["Exif.Image.ImageLength"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.ImageWidth")) != exif_data.end()) {
    display_metadata.width = exif_data["Exif.Image.ImageWidth"].toUint32();
  }
  if (exif_data.findKey(Exiv2::ExifKey("Exif.Image.DateTime")) != exif_data.end()) {
    display_metadata.date_time_str = exif_data["Exif.Image.DateTime"].toString();
    if (display_metadata.date_time_str.size() >= 10) {
      display_metadata.date_time_str[4] =
          '-';  // Change from "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DD HH:MM:SS"
      display_metadata.date_time_str[7] = '-';
    }
  }
}
/**
 * @brief A callback used to parse the basic information of a image file
 *
 * @param buffer
 * @param file_path
 * @param result
 * @param id
 * @param promise
 */
void MetadataDecoder::Decode(std::vector<char> buffer, std::filesystem::path file_path,
                             std::shared_ptr<BufferQueue> result, image_id_t id,
                             std::shared_ptr<std::promise<image_id_t>> promise) {
  try {
    std::shared_ptr<Image> img =
        std::make_shared<Image>(id, file_path, file_path.filename().wstring(), ImageType::DEFAULT);

    img->_exif_data = Exiv2::ImageFactory::open((Exiv2::byte*)buffer.data(), buffer.size());
    img->_exif_data->readMetadata();
    img->_has_exif = !img->_exif_data->exifData().empty();

    GetDisplayMetadataFromExif(img->_exif_data->exifData(), img->_exif_display);

    result->push(img);
    promise->set_value(id);

    return;
  } catch (std::exception& e) {
    // TODO: Append error message to log
    std::cout << e.what() << std::endl;
  }
  // If it fails to read metadata, produce a plain image with minimum metadata
  std::shared_ptr<Image> img =
      std::make_shared<Image>(id, file_path, file_path.filename().wstring(), ImageType::DEFAULT);
  result->push(img);
  promise->set_value(id);
}

void MetadataDecoder::Decode(std::vector<char> buffer, std::shared_ptr<Image> source_img,
                             std::shared_ptr<BufferQueue>              result,
                             std::shared_ptr<std::promise<image_id_t>> promise) {
  try {
    source_img->_exif_data = Exiv2::ImageFactory::open((Exiv2::byte*)buffer.data(), buffer.size());
    source_img->_exif_data->readMetadata();
    source_img->_has_exif = !source_img->_exif_data->exifData().empty();
    GetDisplayMetadataFromExif(source_img->_exif_data->exifData(), source_img->_exif_display);
    result->push(source_img);
    promise->set_value(source_img->_image_id);

    return;
  } catch (std::exception& e) {
    // TODO: Append error message to log
    std::cout << e.what() << std::endl;
  }
  // If it fails to read metadata, produce a plain image with minimum metadata
  result->push(source_img);
  promise->set_value(source_img->_image_id);
}

};  // namespace puerhlab