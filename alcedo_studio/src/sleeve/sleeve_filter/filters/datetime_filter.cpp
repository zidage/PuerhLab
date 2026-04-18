//  Copyright 2025 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#include "sleeve/sleeve_filter/filters/datetime_filter.hpp"

#include <json.hpp>

namespace alcedo {
using json = nlohmann::json;
auto DatetimeFilter::ToJSON() -> std::wstring {
  json o{{"start_time", start_time_}, {"end_time", end_time_}};
  return o;
}
};  // namespace alcedo