#pragma once

#include <ctime>
#include <string>

#include "sleeve_filter.hpp"
#include "type/type.hpp"
#include "value_filter.hpp"

namespace puerhlab {

class FilterableMetadata {
 public:
  // libraw_iparams_t
  std::string    make_           = "";
  std::string    model_          = "";

  // libraw_image_size_t
  unsigned short height_         = 0;
  unsigned short width_          = 0;

  // libraw_lensinfo_t
  // libraw_makernotes_lens_t
  std::string    lens_           = "";
  std::string    lens_make_      = "";
  float          aperture_       = 0.0f;
  float          focal_          = 0.0f;
  bool           has_attachment_ = false;  // for adapter etc.
};

class ExifFilter : public ValueFilter<FilterableMetadata> {
 private:
  FilterableMetadata metadata_;

 public:
  FilterType   type_ = FilterType::EXIF;
  ElementOrder order_;
  void         SetFilter(FilterableMetadata metadata, ElementOrder order);
  void         ResetFilter();
  auto         GetPredicate() -> std::wstring;
  auto         ToJSON() -> std::wstring;
};
};  // namespace puerhlab