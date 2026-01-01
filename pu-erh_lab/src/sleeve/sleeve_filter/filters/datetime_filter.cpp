//  Copyright 2025 Yurun Zi
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "sleeve/sleeve_filter/filters/datetime_filter.hpp"

#include <json.hpp>

namespace puerhlab {
using json = nlohmann::json;
auto DatetimeFilter::ToJSON() -> std::wstring {
  json o{{"start_time", start_time_}, {"end_time", end_time_}};
  return o;
}
};  // namespace puerhlab