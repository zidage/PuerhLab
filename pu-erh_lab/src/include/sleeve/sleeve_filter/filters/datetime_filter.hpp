#pragma once

#include <ctime>
#include <string>

#include "range_filter.hpp"
#include "sleeve_filter.hpp"
#include "type/type.hpp"

namespace puerhlab {
class DatetimeFilter : public RangeFilter<std::time_t> {
 private:
  std::time_t start_time;
  std::time_t end_time;

  void        SetFilter(std::time_t start_time, std::time_t end_time);
  void        ResetFilter();
  auto        GetPredicate() -> std::wstring;
  auto        Hash() -> hash_t;
};
};  // namespace puerhlab