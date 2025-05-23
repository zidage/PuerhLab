#pragma once

#include "sleeve_filter.hpp"

namespace puerhlab {
template <typename T>
class ValueFilter {
 public:
  virtual void SetValue(T value)              = 0;
  virtual void ResetFilter()                  = 0;
  virtual auto GetPredicate() -> std::wstring = 0;
};
};  // namespace puerhlab