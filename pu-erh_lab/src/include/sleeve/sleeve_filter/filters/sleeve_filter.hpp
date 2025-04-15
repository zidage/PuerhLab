#pragma once

#include <cstdint>
#include <string>

#include "type/type.hpp"

namespace puerhlab {

enum class FilterType { Datetime, Exif };
enum class ElementOrder { ASC, DESC };

template <typename T, FilterType>
class SleeveFilter {
 public:
  virtual void   ResetFilter()                  = 0;
  virtual auto   GetPredicate() -> std::wstring = 0;
  virtual hash_t Hash()                         = 0;
};
};  // namespace puerhlab