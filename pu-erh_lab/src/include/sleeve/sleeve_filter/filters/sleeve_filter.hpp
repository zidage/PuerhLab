#pragma once

#include <cstdint>
#include <string>

#include "type/type.hpp"

namespace puerhlab {

enum class ElementOrder { ASC, DESC };

enum class FilterType { DATETIME, EXIF, DEFAULT };
class SleeveFilter {
 public:
  FilterType   _type;
  virtual void ResetFilter()                      = 0;
  virtual auto GetPredicate() -> std::wstring     = 0;
  virtual auto ToJSON() -> std::wstring           = 0;
  virtual void FromJSON(const std::wstring j_str) = 0;
};
};  // namespace puerhlab