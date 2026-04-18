//  Copyright 2025-2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <ctime>
#include <string>

#include "range_filter.hpp"
#include "sleeve_filter.hpp"
#include "type/type.hpp"

namespace alcedo {
class DatetimeFilter : public RangeFilter<std::time_t> {
 public:
  FilterType type_ = FilterType::DATETIME;

  void       SetFilter(std::time_t start_time, std::time_t end_time);
  void       ResetFilter();
  auto       GetPredicate() -> std::wstring;
  auto       ToJSON() -> std::wstring;
  void       FromJSON(const std::wstring j_str);

 private:
  std::time_t start_time_;
  std::time_t end_time_;
};
};  // namespace alcedo