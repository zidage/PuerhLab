#pragma once

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T>
class ValueFilter {
  virtual void SetValue(T value) = 0;
};
};  // namespace puerhlab