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
};
};  // namespace puerhlab