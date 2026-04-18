//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "sleeve/sleeve_filter/filters/exif_filter.hpp"

#include <json.hpp>
#include <string>

namespace alcedo {
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
  json o{{"make", metadata_.make_},
         {"model", metadata_.model_},
         {"height", metadata_.height_},
         {"width", metadata_.width_},
         {"lens", metadata_.lens_},
         {"lens_make", metadata_.lens_make_},
         {"aperture", metadata_.aperture_},
         {"focal", metadata_.focal_},
         {"has_attachment", metadata_.has_attachment_}};
  return o;
}
};  // namespace alcedo