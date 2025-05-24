#pragma once

#include <string>

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T>
class RangeFilter : public SleeveFilter {
 public:
  virtual void SetRange(T range_low, T range_high) = 0;
};
};  // namespace puerhlab