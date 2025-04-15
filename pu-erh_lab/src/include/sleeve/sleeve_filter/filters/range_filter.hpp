#pragma once

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T, FilterType U>
class RangeFilter : public SleeveFilter<T, U> {
  virtual void SetFilter(T range_low, T range_high) = 0;
};
};  // namespace puerhlab