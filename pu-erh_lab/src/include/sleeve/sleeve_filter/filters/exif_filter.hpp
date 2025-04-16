#pragma once

#include <ctime>
#include <string>

#include "type/type.hpp"
#include "value_filter.hpp"

namespace puerhlab {

class FilterableMetadata {
 public:
  // libraw_iparams_t
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
  bool           has_attachment = false;  // for adapter etc.
};

class ExifFilter : public ValueFilter<FilterableMetadata> {
 private:
  FilterableMetadata _metadata;

 public:
  void SetFilter(FilterableMetadata metadata);
  void ResetFilter();
  auto GetPredicate() -> std::wstring;
  auto Hash() -> hash_t;
};
};  // namespace puerhlab