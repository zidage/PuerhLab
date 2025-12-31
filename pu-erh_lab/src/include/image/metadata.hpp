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
#include <cstdint>
#include <json.hpp>
#include <sstream>
#include <string>

namespace puerhlab {
class ExifDisplayMetaData {
 public:
  // Model
  std::string         make          = "";
  std::string         model         = "";
  std::string         lens          = "";
  std::string         lens_make     = "";

  std::string         date_time_str = "";

  // Size
  uint32_t            height        = 0;
  uint32_t            width         = 0;
  uint32_t            image_size    = 0;

  // Technical
  float               aperture      = 0.0f;
  std::pair<int, int> shutter_speed = {0, 0};
  uint64_t            iso           = 0;
  float               focal         = 0.0f;

  // Other
  int                 rating        = 0;

  ExifDisplayMetaData()             = default;
  ExifDisplayMetaData(nlohmann::json exif_json);

  void        ExtractFromJson(nlohmann::json exif_json);

  std::string ToString() const {
    // DEBUG ONLY
    std::stringstream ss;
    // Model
    ss << "Make: " << make << "\n";
    ss << "Model: " << model << "\n";
    ss << "Lens: " << lens << "\n";
    ss << "Lens Make: " << lens_make << "\n";

    // Technical
    ss << "Aperture: " << aperture << "\n";
    ss << "Focal Length: " << focal << " mm\n";
    ss << "Shutter Speed: " << shutter_speed.first << "/" << shutter_speed.second << " s\n";
    ss << "Image Size: " << width << " x " << height << "\n";
    ss << "Date Time: " << date_time_str << "\n";
    ss << "Rating: " << rating << "\n";
    return ss.str();
  }

  auto ToJson() const -> nlohmann::json {
    nlohmann::json exif_json;
    // Model
    exif_json["Make"]           = make;
    exif_json["Model"]          = model;
    exif_json["Lens"]           = lens;
    exif_json["LensMake"]       = lens_make;

    // Technical
    exif_json["Aperture"]       = aperture;
    exif_json["FocalLength"]    = focal;
    exif_json["ISO"]            = iso;
    exif_json["ShutterSpeed"]   = {shutter_speed.first, shutter_speed.second};
    exif_json["ImageHeight"]    = height;
    exif_json["ImageWidth"]     = width;
    exif_json["ImageSize"]      = width * height;
    exif_json["DateTimeString"] = date_time_str;

    // Other
    exif_json["Rating"]         = rating;
    return exif_json;
  }

  void FromJson(const nlohmann::json& exif_json) {
    // Model
    make          = exif_json.value("Make", "");
    model         = exif_json.value("Model", "");
    lens          = exif_json.value("Lens", "");
    lens_make     = exif_json.value("LensMake", "");

    // Technical
    aperture      = exif_json.value("Aperture", 0.0f);
    focal         = exif_json.value("FocalLength", 0.0f);
    iso           = exif_json.value("ISO", 0);
    auto shutter  = exif_json.value("ShutterSpeed", std::vector<int>{0, 0});
    height        = exif_json.value("ImageHeight", 0);
    width         = exif_json.value("ImageWidth", 0);
    image_size    = exif_json.value("ImageSize", 0);

    // Time
    date_time_str = exif_json.value("DateTimeString", "");
    // Other
    rating        = exif_json.value("Rating", 0);
  }
};
};  // namespace puerhlab