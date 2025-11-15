#include <cstdint>
#include <json.hpp>
#include <sstream>
#include <string>

namespace puerhlab {
class ExifDisplayMetaData {
 public:
  std::string make           = "";
  std::string model          = "";

  std::string date_time_str  = "";

  // libraw_image_size_t
  uint32_t    height         = 0;
  uint32_t    width          = 0;

  // libraw_lensinfo_t
  // libraw_makernotes_lens_t
  std::string lens           = "";
  std::string lens_make      = "";
  std::pair<int, int>       aperture       = {0, 0};
  float       focal          = 0.0f;
  bool        has_attachment = false;

  float       maximum        = 0;

  ExifDisplayMetaData()      = default;
  ExifDisplayMetaData(nlohmann::json exif_json);

  void ExtractFromJson(nlohmann::json exif_json);

  std::string ToString() const {
    std::stringstream ss;
    ss << "Make: " << make << "\n";
    ss << "Model: " << model << "\n";
    ss << "Lens: " << lens << "\n";
    ss << "Lens Make: " << lens_make << "\n";
    ss << "Aperture: " << aperture.first << "/" << aperture.second << "\n";
    ss << "Focal Length: " << focal << " mm\n";
    ss << "Image Size: " << width << " x " << height << "\n";
    ss << "Date Time: " << date_time_str << "\n";
    return ss.str();
  }

  auto ToJson() const -> nlohmann::json {
    nlohmann::json exif_json;
    exif_json["Make"]           = make;
    exif_json["Model"]          = model;
    exif_json["Lens"]           = lens;
    exif_json["LensMake"]       = lens_make;
    exif_json["Aperture"]       = {aperture.first, aperture.second};
    exif_json["FocalLength"]    = focal;
    exif_json["ImageHeight"]    = height;
    exif_json["ImageWidth"]     = width;
    exif_json["DateTimeString"] = date_time_str;
    return exif_json;
  }

  void FromJson(const nlohmann::json& exif_json) {
    make           = exif_json.value("Make", "");
    model          = exif_json.value("Model", "");
    lens           = exif_json.value("Lens", "");
    lens_make      = exif_json.value("LensMake", "");
    if (exif_json.contains("Aperture") && exif_json["Aperture"].is_array() &&
        exif_json["Aperture"].size() == 2) {
      aperture.first  = exif_json["Aperture"][0].get<int>();
      aperture.second = exif_json["Aperture"][1].get<int>();
    }
    focal          = exif_json.value("FocalLength", 0.0f);
    height         = exif_json.value("ImageHeight", 0);
    width          = exif_json.value("ImageWidth", 0);
    date_time_str  = exif_json.value("DateTimeString", "");
  }
};
};  // namespace puerhlab