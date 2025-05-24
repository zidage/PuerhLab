#include "sleeve/sleeve_filter/filters/exif_filter.hpp"

#include <json.hpp>
#include <string>

namespace puerhlab {
using json = nlohmann::json;
auto ExifFilter::ToJSON() -> std::wstring {
  //   std::string    make           = "";
  //   std::string    model          = "";

  //   // libraw_image_size_t
  //   unsigned short height         = 0;
  //   unsigned short width          = 0;

  //   // libraw_lensinfo_t
  //   // libraw_makernotes_lens_t
  //   std::string    lens           = "";
  //   std::string    lens_make      = "";
  //   float          aperture       = 0.0f;
  //   float          focal          = 0.0f;
  //   bool           has_attachment = false;  // for adapter etc.
  json o{{"make", _metadata.make},         {"model", _metadata.model}, {"height", _metadata.height},
         {"width", _metadata.width},       {"lens", _metadata.lens},   {"lens_make", _metadata.lens_make},
         {"aperture", _metadata.aperture}, {"focal", _metadata.focal}, {"has_attachment", _metadata.has_attachment}};
  return o;
}
};  // namespace puerhlab