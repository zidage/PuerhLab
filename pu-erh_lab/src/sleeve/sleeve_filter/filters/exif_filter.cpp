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
};  // namespace puerhlab