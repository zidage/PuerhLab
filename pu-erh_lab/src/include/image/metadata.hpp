#include <string>

namespace puerhlab {
struct ExifMetaData {
  std::string    make           = "";
  std::string    model          = "";

  // libraw_image_size_t
  unsigned short height         = 0;
  unsigned short width          = 0;

  // libraw_lensinfo_t
  // libraw_makernotes_lens_t
  std::string    lens           = "";
  std::string    lens_make      = "";
  float          aperture       = 0.0f;
  float          focal          = 0.0f;
  bool           has_attachment = false;
};
};  // namespace puerhlab