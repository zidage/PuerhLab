#pragma once

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T>
class RangeFilter : public SleeveFilter {
 public:
  virtual void   SetRange(T range_low, T range_high) = 0;
  virtual void   ResetFilter()                       = 0;
  virtual auto   GetPredicate() -> std::wstring      = 0;
  virtual hash_t Hash()                              = 0;
};
};  // namespace puerhlab