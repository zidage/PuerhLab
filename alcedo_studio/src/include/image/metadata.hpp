//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once
#include <cstdint>
#include <json.hpp>
#include <sstream>
#include <string>

namespace alcedo {
class ExifDisplayMetaData {
 public:
  // Model
  std::string         make_          = "";
  std::string         model_         = "";
  std::string         lens_          = "";
  std::string         lens_make_     = "";

  std::string         date_time_str_ = "";

  // Size
  uint32_t            height_        = 0;
  uint32_t            width_         = 0;
  uint32_t            image_size_    = 0;

  // Technical
  float               aperture_      = 0.0f;
  std::pair<int, int> shutter_speed_ = {0, 0};
  uint64_t            iso_           = 0;
  float               focal_         = 0.0f;
  float               focal_35mm_    = 0.0f;
  float               focus_distance_m_ = 0.0f;

  // Other
  int                 rating_        = 0;

  ExifDisplayMetaData()             = default;
  ExifDisplayMetaData(nlohmann::json exif_json);

  void        ExtractFromJson(nlohmann::json exif_json);

  std::string ToString() const {
    // DEBUG ONLY
    std::stringstream ss;
    // Model
    ss << "Make: " << make_ << "\n";
    ss << "Model: " << model_ << "\n";
    ss << "Lens: " << lens_ << "\n";
    ss << "Lens Make: " << lens_make_ << "\n";

    // Technical
    ss << "Aperture: " << aperture_ << "\n";
    ss << "Focal Length: " << focal_ << " mm\n";
    ss << "Focal Length (35mm eq): " << focal_35mm_ << " mm\n";
    ss << "Focus Distance: " << focus_distance_m_ << " m\n";
    ss << "Shutter Speed: " << shutter_speed_.first << "/" << shutter_speed_.second << " s\n";
    ss << "Image Size: " << width_ << " x " << height_ << "\n";
    ss << "Date Time: " << date_time_str_ << "\n";
    ss << "Rating: " << rating_ << "\n";
    return ss.str();
  }

  auto ToJson() const -> nlohmann::json {
    nlohmann::json exif_json;
    // Model
    exif_json["Make"]           = make_;
    exif_json["Model"]          = model_;
    exif_json["Lens"]           = lens_;
    exif_json["LensMake"]       = lens_make_;

    // Technical
    exif_json["Aperture"]       = aperture_;
    exif_json["FocalLength"]    = focal_;
    exif_json["FocalLength35mm"] = focal_35mm_;
    exif_json["FocusDistanceM"] = focus_distance_m_;
    exif_json["ISO"]            = iso_;
    exif_json["ShutterSpeed"]   = {shutter_speed_.first, shutter_speed_.second};
    exif_json["ImageHeight"]    = height_;
    exif_json["ImageWidth"]     = width_;
    exif_json["ImageSize"]      = width_ * height_;
    exif_json["DateTimeString"] = date_time_str_;

    // Other
    exif_json["Rating"]         = rating_;
    return exif_json;
  }

  void FromJson(const nlohmann::json& exif_json) {
    // Model
    make_          = exif_json.value("Make", "");
    model_         = exif_json.value("Model", "");
    lens_          = exif_json.value("Lens", "");
    lens_make_     = exif_json.value("LensMake", "");

    // Technical
    aperture_      = exif_json.value("Aperture", 0.0f);
    focal_         = exif_json.value("FocalLength", 0.0f);
    focal_35mm_    = exif_json.value("FocalLength35mm", 0.0f);
    focus_distance_m_ = exif_json.value("FocusDistanceM", 0.0f);
    iso_           = exif_json.value("ISO", 0);
    auto shutter  = exif_json.value("ShutterSpeed", std::vector<int>{0, 0});
    height_        = exif_json.value("ImageHeight", 0);
    width_         = exif_json.value("ImageWidth", 0);
    image_size_    = exif_json.value("ImageSize", 0);

    // Time
    date_time_str_ = exif_json.value("DateTimeString", "");
    // Other
    rating_        = exif_json.value("Rating", 0);
  }
};
};  // namespace alcedo
