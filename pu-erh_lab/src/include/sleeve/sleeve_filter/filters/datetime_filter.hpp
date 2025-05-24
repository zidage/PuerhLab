#pragma once

#include <ctime>
#include <string>

#include "range_filter.hpp"
#include "sleeve_filter.hpp"
#include "type/type.hpp"

namespace puerhlab {
class DatetimeFilter : public RangeFilter<std::time_t> {
 public:
  FilterType _type = FilterType::DATETIME;

  void       SetFilter(std::time_t start_time, std::time_t end_time);
  void       ResetFilter();
  auto       GetPredicate() -> std::wstring;
  auto       ToJSON() -> std::wstring;
  void       FromJSON(const std::wstring j_str);

 private:
  std::time_t _start_time;
  std::time_t _end_time;
};
};  // namespace puerhlab